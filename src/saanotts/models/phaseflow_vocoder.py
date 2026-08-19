"""LACE PhaseFlow vocoder: split magnitude/phase experts under 700k params.

The decoder retains a neural complex-spectral synthesis path, but does not ask
a small shared trunk to learn independent wrapped phase at every frame. A
gradient-isolated phase expert predicts residual phase flow around the expected
FFT-bin advance. Explicit voicing and aperiodicity controls gate that advance,
while a separate expert preserves magnitude capacity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class PhaseFlowVocoderConfig:
    mel_channels: int = 100
    prosody_channels: int = 3
    width: int = 160
    shared_depth: int = 3
    expert_depth: int = 1
    expansion: int = 3
    pointwise_rank: int = 62
    kernel_size: int = 7
    noise_channels: int = 4
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    sample_rate: int = 24_000

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LowRankConvNeXtBlock1d(nn.Module):
    """Full-width ConvNeXt topology with factorized pointwise matrices."""

    def __init__(
        self,
        width: int,
        expansion: int,
        rank: int,
        kernel_size: int,
        scale: float,
    ) -> None:
        super().__init__()
        expanded = width * expansion
        self.depthwise = nn.Conv1d(
            width,
            width,
            kernel_size,
            padding=kernel_size // 2,
            groups=width,
        )
        self.norm = nn.LayerNorm(width, eps=1e-6)
        self.up_in = nn.Linear(width, rank, bias=False)
        self.up_out = nn.Linear(rank, expanded)
        self.down_in = nn.Linear(expanded, rank, bias=False)
        self.down_out = nn.Linear(rank, width)
        self.scale = nn.Parameter(torch.full((width,), float(scale)))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = value
        value = self.depthwise(value).transpose(1, 2)
        value = self.norm(value)
        value = self.up_out(self.up_in(value))
        value = F.gelu(value)
        value = self.down_out(self.down_in(value))
        return residual + (self.scale * value).transpose(1, 2)


class PhaseFlowVocoder(nn.Module):
    """Mel/prosody to waveform decoder with gated phase-flow integration."""

    def __init__(self, config: PhaseFlowVocoderConfig = PhaseFlowVocoderConfig()) -> None:
        super().__init__()
        if config.kernel_size <= 0 or config.kernel_size % 2 == 0:
            raise ValueError("PhaseFlow kernel size must be positive and odd")
        if config.shared_depth <= 0 or config.expert_depth <= 0:
            raise ValueError("PhaseFlow shared and expert depths must be positive")
        if config.win_length != config.n_fft:
            raise ValueError("PhaseFlow currently requires win_length == n_fft")
        self.config = config
        input_channels = config.mel_channels + config.prosody_channels
        self.embed = nn.Conv1d(
            input_channels,
            config.width,
            config.kernel_size,
            padding=config.kernel_size // 2,
        )
        self.noise_adapter = nn.Conv1d(
            config.noise_channels,
            config.width,
            config.kernel_size,
            padding=config.kernel_size // 2,
        )
        self.input_norm = nn.LayerNorm(config.width, eps=1e-6)
        total_blocks = config.shared_depth + 2 * config.expert_depth
        scale = 1.0 / float(total_blocks)
        self.shared = nn.ModuleList(
            LowRankConvNeXtBlock1d(
                config.width,
                config.expansion,
                config.pointwise_rank,
                config.kernel_size,
                scale,
            )
            for _ in range(config.shared_depth)
        )
        self.magnitude_expert = nn.ModuleList(
            LowRankConvNeXtBlock1d(
                config.width,
                config.expansion,
                config.pointwise_rank,
                config.kernel_size,
                scale,
            )
            for _ in range(config.expert_depth)
        )
        self.phase_expert = nn.ModuleList(
            LowRankConvNeXtBlock1d(
                config.width,
                config.expansion,
                config.pointwise_rank,
                config.kernel_size,
                scale,
            )
            for _ in range(config.expert_depth)
        )
        self.final_norm = nn.LayerNorm(config.width, eps=1e-6)
        bins = config.n_fft // 2 + 1
        self.magnitude_head = nn.Linear(config.width, bins)
        self.phase_flow_head = nn.Linear(config.width, bins)
        self.continuity_head = nn.Linear(config.width, 1)

        advance = (
            2.0
            * math.pi
            * config.hop_length
            * torch.arange(bins, dtype=torch.float32)
            / float(config.n_fft)
        )
        self.register_buffer("bin_phase_advance", advance, persistent=True)
        self.register_buffer("dc_impulse", self._dc_block_impulse(), persistent=False)
        self.apply(self._initialize)
        nn.init.zeros_(self.noise_adapter.weight)
        nn.init.zeros_(self.noise_adapter.bias)
        nn.init.zeros_(self.continuity_head.weight)
        nn.init.constant_(self.continuity_head.bias, 2.0)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @staticmethod
    def _dc_block_impulse(taps: int = 4096, radius: float = 0.995) -> torch.Tensor:
        impulse = torch.zeros(taps, dtype=torch.float64)
        impulse[0] = 1.0
        index = torch.arange(1, taps, dtype=torch.float64)
        impulse[1:] = (radius - 1.0) * radius ** (index - 1.0)
        return impulse.to(dtype=torch.float32)

    def _validate_inputs(
        self,
        mel: torch.Tensor,
        prosody: torch.Tensor,
        noise: torch.Tensor | None,
    ) -> torch.Tensor:
        expected_mel = (mel.shape[0], self.config.mel_channels, mel.shape[-1])
        expected_prosody = (mel.shape[0], self.config.prosody_channels, mel.shape[-1])
        if tuple(mel.shape) != expected_mel:
            raise ValueError(f"expected mel {expected_mel}, got {tuple(mel.shape)}")
        if tuple(prosody.shape) != expected_prosody:
            raise ValueError(f"expected prosody {expected_prosody}, got {tuple(prosody.shape)}")
        if noise is None:
            noise = torch.randn(
                mel.shape[0],
                self.config.noise_channels,
                mel.shape[-1],
                device=mel.device,
                dtype=mel.dtype,
            )
        expected_noise = (mel.shape[0], self.config.noise_channels, mel.shape[-1])
        if tuple(noise.shape) != expected_noise:
            raise ValueError(f"expected noise {expected_noise}, got {tuple(noise.shape)}")
        return noise

    def forward(
        self,
        mel: torch.Tensor,
        prosody: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        noise = self._validate_inputs(mel, prosody, noise)
        value = self.embed(torch.cat([mel, prosody], dim=1)) + self.noise_adapter(noise)
        value = self.input_norm(value.transpose(1, 2)).transpose(1, 2)
        for block in self.shared:
            value = block(value)

        magnitude_state = value
        for block in self.magnitude_expert:
            magnitude_state = block(magnitude_state)
        # Phase reconstruction must not consume the shared envelope capacity.
        phase_state = value.detach()
        for block in self.phase_expert:
            phase_state = block(phase_state)

        magnitude_features = self.final_norm(magnitude_state.transpose(1, 2))
        phase_features = self.final_norm(phase_state.transpose(1, 2))
        log_magnitude = self.magnitude_head(magnitude_features).transpose(1, 2)
        log_magnitude = torch.clamp(log_magnitude, min=-12.0, max=math.log(100.0))
        phase_coordinate = math.pi * torch.tanh(
            self.phase_flow_head(phase_features).transpose(1, 2)
        )

        learned_continuity = torch.sigmoid(
            self.continuity_head(phase_features).transpose(1, 2)
        )
        voicing = prosody[:, 1:2].clamp(0.0, 1.0)
        aperiodicity = prosody[:, 2:3].clamp(0.0, 1.0)
        continuity = learned_continuity * voicing * (1.0 - 0.5 * aperiodicity)
        phase = phase_coordinate.clone()
        if phase.shape[-1] > 1:
            increments = (
                continuity[..., 1:] * self.bin_phase_advance[None, :, None]
                + phase_coordinate[..., 1:]
            )
            phase[..., 1:] = phase_coordinate[..., :1] + torch.cumsum(increments, dim=-1)

        magnitude = torch.exp(log_magnitude)
        mask = torch.ones_like(magnitude)
        mask[:, 0] = 0.0
        mask[:, -1] = 0.0
        spectrum = magnitude * mask * torch.polar(torch.ones_like(phase), phase)
        return {
            "spectrum": spectrum,
            "log_magnitude": log_magnitude,
            "phase": phase,
            "phase_coordinate": phase_coordinate,
            "continuity": continuity,
        }

    def _dc_block(self, waveform: torch.Tensor) -> torch.Tensor:
        samples = int(waveform.shape[-1])
        linear = samples + int(self.dc_impulse.numel()) - 1
        fft_size = 1 << (linear - 1).bit_length()
        response = torch.fft.rfft(self.dc_impulse.to(waveform), fft_size)
        filtered = torch.fft.irfft(torch.fft.rfft(waveform, fft_size) * response, fft_size)
        return filtered[..., :samples]

    def synthesize(self, spectrum: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
        waveform = torch.istft(
            spectrum,
            self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            window=window,
            center=True,
        )
        return self._dc_block(waveform)


def phaseflow_parameter_count(model: PhaseFlowVocoder) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


__all__ = [
    "LowRankConvNeXtBlock1d",
    "PhaseFlowVocoder",
    "PhaseFlowVocoderConfig",
    "phaseflow_parameter_count",
]
