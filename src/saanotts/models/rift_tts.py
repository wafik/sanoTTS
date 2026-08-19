#!/usr/bin/env python3
"""RIFT-TTS: a recurrent-depth, phase-anchored text-to-wave model.

RIFT reuses one convolutional transition cell before and after duration
expansion.  The output head predicts low-rank spectral dictionary coordinates
and reconstructs audio with a deterministic same-padded iSTFT.  Phase is a
bounded correction of an integrated instantaneous-frequency oscillator,
which preserves continuity without asking the network to regress every frame's
wrapped phase independently.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


SAMPLE_RATE = 24_000
N_FFT = 1_024
HOP_LENGTH = 256
N_MELS = 80


@dataclass(frozen=True)
class RiftConfig:
    vocab_size: int = 62
    width: int = 96
    token_steps: int = 4
    frame_steps: int = 6
    kernel_size: int = 7
    expansion: int = 4
    magnitude_rank: int = 32
    if_rank: int = 24
    anchor_rank: int = 24
    gate_rank: int = 8
    min_duration: int = 2
    max_duration: int = 80
    phase_trunk_detach: bool = True
    sample_rate: int = SAMPLE_RATE
    n_fft: int = N_FFT
    hop_length: int = HOP_LENGTH
    n_mels: int = N_MELS

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RiftFrontConfig:
    vocab_size: int = 62
    width: int = 64
    token_steps: int = 4
    frame_steps: int = 6
    kernel_size: int = 7
    expansion: int = 2
    n_mels: int = 100
    mel_residual_rank: int = 0
    mel_residual_kernel_size: int = 3
    min_duration: int = 2
    max_duration: int = 80

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SharedLoopCell(nn.Module):
    """One residual transition reused on both sides of length regulation."""

    def __init__(self, width: int, kernel_size: int, expansion: int) -> None:
        super().__init__()
        if width <= 0 or kernel_size <= 0 or kernel_size % 2 == 0 or expansion <= 0:
            raise ValueError("invalid shared-loop dimensions")
        hidden = width * expansion
        self.depthwise = nn.Conv1d(
            width,
            width,
            kernel_size,
            padding=kernel_size // 2,
            groups=width,
        )
        self.norm = nn.GroupNorm(1, width)
        self.up = nn.Conv1d(width, hidden, 1)
        self.down = nn.Conv1d(hidden, width, 1)
        self.residual_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, value: torch.Tensor, loop_bias: torch.Tensor) -> torch.Tensor:
        residual = self.depthwise(value)
        residual = self.norm(residual) + loop_bias
        residual = self.down(F.gelu(self.up(residual)))
        return value + self.residual_scale * residual


class LowRankDictionaryHead(nn.Module):
    """Frame coefficients times a learned output dictionary."""

    def __init__(self, in_channels: int, rank: int, out_channels: int) -> None:
        super().__init__()
        if in_channels <= 0 or rank <= 0 or out_channels <= 0:
            raise ValueError("dictionary dimensions must be positive")
        self.coefficients = nn.Conv1d(in_channels, rank, 1)
        self.dictionary = nn.Parameter(torch.empty(rank, out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.normal_(self.dictionary, mean=0.0, std=0.02)
        nn.init.normal_(self.coefficients.weight, mean=0.0, std=0.02)
        if self.coefficients.bias is not None:
            nn.init.zeros_(self.coefficients.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        coefficients = self.coefficients(value)
        return torch.einsum("brt,ro->bot", coefficients, self.dictionary) + self.bias[None, :, None]


def wrap_phase(value: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(value), torch.cos(value))


class RiftTTS(nn.Module):
    """A single-voice phoneme-to-wave RIFT synthesizer.

    Input is a batch of one phoneme-ID sequence.  Teacher durations may be
    supplied during training; when omitted, the model's own duration head and
    hard length regulator are used.
    """

    def __init__(self, config: RiftConfig) -> None:
        super().__init__()
        if config.vocab_size <= 0 or config.width <= 0:
            raise ValueError("vocab_size and width must be positive")
        if config.token_steps <= 0 or config.frame_steps <= 0:
            raise ValueError("token_steps and frame_steps must be positive")
        if config.n_fft <= 0 or config.n_fft % 2 != 0:
            raise ValueError("n_fft must be a positive even integer")
        if config.hop_length <= 0 or config.hop_length >= config.n_fft:
            raise ValueError("hop_length must be positive and smaller than n_fft")
        if config.min_duration <= 0 or config.max_duration < config.min_duration:
            raise ValueError("duration bounds are invalid")
        self.config = config
        width = config.width
        bins = config.n_fft // 2 + 1
        loop_steps = config.token_steps + config.frame_steps

        self.embedding = nn.Embedding(config.vocab_size, width)
        self.token_input = nn.Conv1d(width + 3, width, 1)
        self.shared_cell = SharedLoopCell(width, config.kernel_size, config.expansion)
        self.loop_bias = nn.Parameter(torch.zeros(loop_steps, width))
        self.stage_bias = nn.Parameter(torch.zeros(2, width))
        self.reinject_logit = nn.Parameter(torch.full((loop_steps,), -2.0))
        self.duration_head = nn.Conv1d(width, 1, 1)
        self.frame_input = nn.Conv1d(width + 4, width, 1)

        self.mel_head = nn.Conv1d(width, config.n_mels, 1)
        self.magnitude_head = LowRankDictionaryHead(width, config.magnitude_rank, bins)
        self.if_head = LowRankDictionaryHead(width, config.if_rank, bins)
        self.anchor_head = LowRankDictionaryHead(width, config.anchor_rank, bins * 2)
        self.phase_gate_head = LowRankDictionaryHead(width, config.gate_rank, bins)

        self.register_buffer("window", torch.hann_window(config.n_fft), persistent=False)
        advance = (
            2.0
            * math.pi
            * config.hop_length
            * torch.arange(bins, dtype=torch.float32)
            / float(config.n_fft)
        )
        self.register_buffer("bin_phase_advance", advance, persistent=True)
        self.reset_synthesis_biases()

    def reset_synthesis_biases(self) -> None:
        with torch.no_grad():
            self.magnitude_head.bias.fill_(-4.0)
            self.if_head.bias.zero_()
            self.anchor_head.bias.zero_()
            bins = self.config.n_fft // 2 + 1
            self.anchor_head.bias[:bins].fill_(1.0)
            self.phase_gate_head.bias.fill_(-2.0)

    def _iterate(self, base: torch.Tensor, *, start: int, count: int, stage: int) -> torch.Tensor:
        value = base
        stage_bias = self.stage_bias[stage][None, :, None]
        for offset in range(count):
            step = start + offset
            gain = torch.sigmoid(self.reinject_logit[step])
            value = value + gain * base
            bias = stage_bias + self.loop_bias[step][None, :, None]
            value = self.shared_cell(value, bias)
        return value

    def encode_tokens(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] <= 0:
            raise ValueError(f"expected ids [1,N] with N > 0, got {tuple(ids.shape)}")
        if int(ids.min().item()) < 0 or int(ids.max().item()) >= self.config.vocab_size:
            raise ValueError("phoneme ID is outside the configured vocabulary")
        tokens = ids.shape[1]
        position = torch.linspace(0.0, 1.0, tokens, device=ids.device, dtype=torch.float32)
        center_distance = torch.abs(position - 0.5) * 2.0
        length_hint = torch.full_like(
            position,
            math.log1p(float(tokens)) / math.log1p(512.0),
        )
        embedding = self.embedding(ids).transpose(1, 2)
        token_features = torch.stack([position, center_distance, length_hint], dim=0).unsqueeze(0)
        base = self.token_input(torch.cat([embedding, token_features.to(embedding.dtype)], dim=1))
        states = self._iterate(base, start=0, count=self.config.token_steps, stage=0)
        log_durations = self.duration_head(states).squeeze(1)
        return states, log_durations

    def durations_from_prediction(self, log_durations: torch.Tensor) -> torch.Tensor:
        predicted = torch.expm1(
            torch.clamp(
                log_durations,
                min=math.log1p(float(self.config.min_duration)),
                max=math.log1p(float(self.config.max_duration)),
            )
        )
        return torch.round(predicted).to(dtype=torch.long).clamp(
            min=self.config.min_duration,
            max=self.config.max_duration,
        )

    def _length_regulate(
        self,
        token_states: torch.Tensor,
        durations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if durations.ndim == 2:
            if durations.shape[0] != 1:
                raise ValueError("RIFT currently supports duration batch size one")
            durations = durations[0]
        if durations.ndim != 1 or durations.numel() != token_states.shape[-1]:
            raise ValueError("duration shape does not match token states")
        durations = durations.to(device=token_states.device, dtype=torch.long)
        if bool(torch.any(durations <= 0)):
            raise ValueError("all durations must be positive")
        frames = int(durations.sum().item())
        expanded = torch.repeat_interleave(token_states[0].transpose(0, 1), durations, dim=0)
        if expanded.shape[0] != frames:
            raise RuntimeError("length regulator produced the wrong frame count")

        global_position = torch.linspace(0.0, 1.0, frames, device=token_states.device)
        token_position_parts: list[torch.Tensor] = []
        local_position_parts: list[torch.Tensor] = []
        duration_parts: list[torch.Tensor] = []
        denom = max(int(durations.numel()) - 1, 1)
        for index, duration_value in enumerate(durations.unbind()):
            duration = int(duration_value.item())
            token_position_parts.append(
                torch.full((duration,), float(index) / float(denom), device=token_states.device)
            )
            local_position_parts.append(
                torch.linspace(0.0, 1.0, duration, device=token_states.device)
            )
            duration_parts.append(
                torch.full(
                    (duration,),
                    math.log1p(float(duration)) / math.log1p(float(self.config.max_duration)),
                    device=token_states.device,
                )
            )
        frame_features = torch.stack(
            [
                global_position,
                torch.cat(token_position_parts),
                torch.cat(local_position_parts),
                torch.cat(duration_parts),
            ],
            dim=0,
        ).unsqueeze(0)
        return expanded.transpose(0, 1).unsqueeze(0), frame_features

    def _spectra(self, frame_states: torch.Tensor) -> dict[str, torch.Tensor]:
        log_magnitude = torch.clamp(self.magnitude_head(frame_states), min=-12.0, max=6.0)
        phase_states = frame_states.detach() if self.config.phase_trunk_detach else frame_states
        if_residual = math.pi * torch.tanh(self.if_head(phase_states))

        anchor_logits = self.anchor_head(phase_states)
        bins = self.config.n_fft // 2 + 1
        anchor_real, anchor_imag = anchor_logits[:, :bins], anchor_logits[:, bins:]
        anchor_norm = torch.sqrt(anchor_real.square() + anchor_imag.square()).clamp_min(1e-6)
        anchor_phase = torch.atan2(anchor_imag / anchor_norm, anchor_real / anchor_norm)

        oscillator_phase = anchor_phase[..., :1]
        if if_residual.shape[-1] > 1:
            increments = self.bin_phase_advance[None, :, None] + if_residual[..., 1:]
            oscillator_phase = torch.cat(
                [oscillator_phase, oscillator_phase + torch.cumsum(increments, dim=-1)],
                dim=-1,
            )
        phase_gate = 0.25 * torch.sigmoid(self.phase_gate_head(phase_states))
        anchor_error = wrap_phase(anchor_phase - oscillator_phase)
        phase = oscillator_phase + phase_gate * anchor_error
        spectrum = torch.polar(torch.exp(log_magnitude), phase)
        return {
            "spectrum": spectrum,
            "log_magnitude": log_magnitude,
            "if_residual": if_residual,
            "anchor_phase": anchor_phase,
            "oscillator_phase": oscillator_phase,
            "phase_gate": phase_gate,
            "phase": phase,
        }

    def same_stft(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.ndim != 2:
            raise ValueError(f"expected audio [B,S], got {tuple(audio.shape)}")
        pad = (self.config.n_fft - self.config.hop_length) // 2
        padded = F.pad(audio.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
        frames = padded.unfold(-1, self.config.n_fft, self.config.hop_length)
        frames = frames * self.window.to(device=audio.device, dtype=audio.dtype)
        return torch.fft.rfft(frames, n=self.config.n_fft, dim=-1).transpose(1, 2)

    def same_istft(self, spectrum: torch.Tensor) -> torch.Tensor:
        bins = self.config.n_fft // 2 + 1
        if spectrum.ndim != 3 or spectrum.shape[1] != bins:
            raise ValueError(f"expected spectrum [B,{bins},T], got {tuple(spectrum.shape)}")
        batch, _, frame_count = spectrum.shape
        window = self.window.to(device=spectrum.device, dtype=spectrum.real.dtype)
        inverse = torch.fft.irfft(spectrum, n=self.config.n_fft, dim=1) * window[None, :, None]
        output_size = (frame_count - 1) * self.config.hop_length + self.config.n_fft
        audio = F.fold(
            inverse,
            output_size=(1, output_size),
            kernel_size=(1, self.config.n_fft),
            stride=(1, self.config.hop_length),
        )[:, 0, 0]
        envelope_frames = window.square().view(1, self.config.n_fft, 1).expand(1, self.config.n_fft, frame_count)
        envelope = F.fold(
            envelope_frames,
            output_size=(1, output_size),
            kernel_size=(1, self.config.n_fft),
            stride=(1, self.config.hop_length),
        )[0, 0, 0]
        pad = (self.config.n_fft - self.config.hop_length) // 2
        audio = audio[:, pad:-pad]
        envelope = envelope[pad:-pad]
        expected = frame_count * self.config.hop_length
        if audio.shape != (batch, expected):
            raise RuntimeError(f"same-iSTFT shape invariant failed: {tuple(audio.shape)} != {(batch, expected)}")
        return audio / envelope.clamp_min(1e-7)

    def acoustic_forward(
        self,
        ids: torch.Tensor,
        durations: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        token_states, log_durations = self.encode_tokens(ids)
        used_durations = (
            self.durations_from_prediction(log_durations)
            if durations is None
            else durations.to(device=ids.device, dtype=torch.long)
        )
        expanded, frame_features = self._length_regulate(token_states, used_durations)
        base = self.frame_input(torch.cat([expanded, frame_features.to(expanded.dtype)], dim=1))
        frame_states = self._iterate(
            base,
            start=self.config.token_steps,
            count=self.config.frame_steps,
            stage=1,
        )
        return {
            "mel": self.mel_head(frame_states),
            "log_durations": log_durations,
            "durations": used_durations,
            "token_states": token_states,
            "frame_states": frame_states,
        }

    def forward(
        self,
        ids: torch.Tensor,
        durations: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        acoustic = self.acoustic_forward(ids, durations)
        result = self._spectra(acoustic["frame_states"])
        result.update(acoustic)
        result["audio"] = self.same_istft(result["spectrum"])
        return result


class RiftAcousticFront(RiftTTS):
    """Duration-plus-mel RIFT front end with no deployed synthesis heads."""

    def __init__(self, config: RiftFrontConfig) -> None:
        parent_config = RiftConfig(
            vocab_size=config.vocab_size,
            width=config.width,
            token_steps=config.token_steps,
            frame_steps=config.frame_steps,
            kernel_size=config.kernel_size,
            expansion=config.expansion,
            magnitude_rank=1,
            if_rank=1,
            anchor_rank=1,
            gate_rank=1,
            min_duration=config.min_duration,
            max_duration=config.max_duration,
            n_mels=config.n_mels,
        )
        super().__init__(parent_config)
        self.front_config = config
        del self.magnitude_head
        del self.if_head
        del self.anchor_head
        del self.phase_gate_head
        if config.mel_residual_rank < 0:
            raise ValueError("mel_residual_rank must be non-negative")
        if config.mel_residual_kernel_size <= 0 or config.mel_residual_kernel_size % 2 == 0:
            raise ValueError("mel_residual_kernel_size must be a positive odd integer")
        if config.mel_residual_rank:
            rank = config.mel_residual_rank
            self.mel_residual_up = nn.Conv1d(config.width, rank, 1)
            self.mel_residual_depthwise = nn.Conv1d(
                rank,
                rank,
                config.mel_residual_kernel_size,
                padding=config.mel_residual_kernel_size // 2,
                groups=rank,
            )
            self.mel_residual_down = nn.Conv1d(rank, config.n_mels, 1)
            nn.init.zeros_(self.mel_residual_down.weight)
            nn.init.zeros_(self.mel_residual_down.bias)

    def forward(
        self,
        ids: torch.Tensor,
        durations: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        result = self.acoustic_forward(ids, durations)
        if self.front_config.mel_residual_rank:
            residual = F.gelu(self.mel_residual_up(result["frame_states"]))
            residual = F.gelu(self.mel_residual_depthwise(residual))
            result["mel"] = result["mel"] + self.mel_residual_down(residual)
        return result


@torch.no_grad()
def add_zero_initialized_front_mel_residual(
    source: RiftAcousticFront,
    *,
    rank: int,
    kernel_size: int = 3,
) -> RiftAcousticFront:
    """Add a temporal low-rank mel correction without changing source output."""

    if source.front_config.mel_residual_rank:
        raise ValueError("source front already has a mel residual")
    if rank <= 0:
        raise ValueError("rank must be positive")
    reference = next(source.parameters())
    target = RiftAcousticFront(
        replace(
            source.front_config,
            mel_residual_rank=rank,
            mel_residual_kernel_size=kernel_size,
        )
    ).to(device=reference.device, dtype=reference.dtype)
    result = target.load_state_dict(source.state_dict(), strict=False)
    expected_missing = {
        "mel_residual_up.weight",
        "mel_residual_up.bias",
        "mel_residual_depthwise.weight",
        "mel_residual_depthwise.bias",
        "mel_residual_down.weight",
        "mel_residual_down.bias",
    }
    if set(result.missing_keys) != expected_missing or result.unexpected_keys:
        raise RuntimeError(f"unexpected residual initialization state mismatch: {result}")
    target.train(source.training)
    return target


@torch.no_grad()
def expand_front_shared_cell_net2wider(
    source: RiftAcousticFront,
    *,
    target_expansion: int,
) -> RiftAcousticFront:
    """Widen the shared pointwise cell while preserving the front's function.

    New hidden units duplicate existing units.  Splitting each duplicated
    unit's outgoing weight between the original and its copy makes the widened
    GELU MLP equivalent at initialization, while both copies remain trainable.
    """

    source_config = source.front_config
    if target_expansion <= source_config.expansion:
        raise ValueError("target_expansion must be greater than the source expansion")
    width = source_config.width
    source_hidden = width * source_config.expansion
    target_hidden = width * target_expansion
    added_hidden = target_hidden - source_hidden
    if added_hidden > source_hidden:
        raise ValueError("Net2Wider expansion cannot add more units than the source cell contains")

    reference = next(source.parameters())
    target = RiftAcousticFront(replace(source_config, expansion=target_expansion)).to(
        device=reference.device,
        dtype=reference.dtype,
    )
    source_state = source.state_dict()
    target_state = target.state_dict()
    resized = {
        "shared_cell.up.weight",
        "shared_cell.up.bias",
        "shared_cell.down.weight",
    }
    for name, value in source_state.items():
        if name in resized:
            continue
        if name not in target_state or target_state[name].shape != value.shape:
            raise RuntimeError(f"unexpected Net2Wider state mismatch for {name}")
        target_state[name].copy_(value)

    source_up_weight = source_state["shared_cell.up.weight"]
    source_up_bias = source_state["shared_cell.up.bias"]
    source_down_weight = source_state["shared_cell.down.weight"]
    target_up_weight = target_state["shared_cell.up.weight"]
    target_up_bias = target_state["shared_cell.up.bias"]
    target_down_weight = target_state["shared_cell.down.weight"]

    target_up_weight[:source_hidden].copy_(source_up_weight)
    target_up_bias[:source_hidden].copy_(source_up_bias)
    target_up_weight[source_hidden:].copy_(source_up_weight[:added_hidden])
    target_up_bias[source_hidden:].copy_(source_up_bias[:added_hidden])
    target_down_weight[:, :source_hidden].copy_(source_down_weight)
    target_down_weight[:, :added_hidden].mul_(0.5)
    target_down_weight[:, source_hidden:].copy_(source_down_weight[:, :added_hidden] * 0.5)

    target.load_state_dict(target_state, strict=True)
    target.train(source.training)
    return target


def count_parameters(module: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters()))


def parameter_breakdown(model: RiftTTS) -> dict[str, int]:
    groups = {
        "token_frontend": [model.embedding, model.token_input, model.duration_head],
        "shared_recurrent_cell": [model.shared_cell],
        "frame_frontend": [model.frame_input],
        "mel_supervision_head": [model.mel_head],
        "magnitude_dictionary": [model.magnitude_head],
        "phase_dictionaries": [model.if_head, model.anchor_head, model.phase_gate_head],
    }
    values = {
        name: int(sum(count_parameters(module) for module in modules))
        for name, modules in groups.items()
    }
    values["loop_and_stage_parameters"] = int(
        model.loop_bias.numel() + model.stage_bias.numel() + model.reinject_logit.numel()
    )
    values["total"] = count_parameters(model)
    return values


def acoustic_front_parameter_breakdown(model: RiftAcousticFront) -> dict[str, int]:
    values = {
        "token_frontend": int(
            count_parameters(model.embedding)
            + count_parameters(model.token_input)
            + count_parameters(model.duration_head)
        ),
        "shared_recurrent_cell": count_parameters(model.shared_cell),
        "frame_frontend": count_parameters(model.frame_input),
        "mel_head": count_parameters(model.mel_head),
        "mel_residual_head": int(
            count_parameters(model.mel_residual_up)
            + count_parameters(model.mel_residual_depthwise)
            + count_parameters(model.mel_residual_down)
            if model.front_config.mel_residual_rank
            else 0
        ),
        "loop_and_stage_parameters": int(
            model.loop_bias.numel() + model.stage_bias.numel() + model.reinject_logit.numel()
        ),
    }
    values["total"] = count_parameters(model)
    return values


__all__ = [
    "HOP_LENGTH",
    "N_FFT",
    "N_MELS",
    "SAMPLE_RATE",
    "LowRankDictionaryHead",
    "RiftAcousticFront",
    "RiftConfig",
    "RiftFrontConfig",
    "RiftTTS",
    "SharedLoopCell",
    "add_zero_initialized_front_mel_residual",
    "count_parameters",
    "expand_front_shared_cell_net2wider",
    "acoustic_front_parameter_breakdown",
    "parameter_breakdown",
    "wrap_phase",
]
