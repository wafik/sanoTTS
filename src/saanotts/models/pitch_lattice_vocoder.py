"""PitchLattice: a sub-700k pitch-memory waveform decoder.

The decoder follows the framewise autoregressive topology that makes FARGAN
effective at small sizes, but adapts it to the 24 kHz / 256-sample sanoTTS
interface.  Its distinctive mechanism is a control-gated pitch lattice: each
64-sample subframe can reuse excitation from both one and two pitch periods in
the past.  This gives periodic speech a phase-stable path that does not require
the network to reconstruct Fourier phase or regenerate every harmonic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class PitchLatticeConfig:
    sample_rate: int = 24_000
    frame_size: int = 256
    subframes: int = 4
    feature_dim: int = 24
    spectral_features: int = 20
    pitch_embedding_dim: int = 12
    pitch_embedding_bins: int = 512
    condition_hidden: int = 64
    condition_conv: int = 128
    subframe_condition: int = 80
    fwc_width: int = 154
    gru1_width: int = 140
    gru2_width: int = 106
    gru3_width: int = 106
    skip_width: int = 106
    memory_size: int = 1_024
    min_period: int = 48
    max_period: int = 480
    quantization_noise: float = 1.0 / 127.0

    @property
    def subframe_size(self) -> int:
        return self.frame_size // self.subframes

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        dimensions = (
            self.sample_rate,
            self.frame_size,
            self.subframes,
            self.feature_dim,
            self.spectral_features,
            self.pitch_embedding_dim,
            self.pitch_embedding_bins,
            self.condition_hidden,
            self.condition_conv,
            self.subframe_condition,
            self.fwc_width,
            self.gru1_width,
            self.gru2_width,
            self.gru3_width,
            self.skip_width,
            self.memory_size,
            self.min_period,
            self.max_period,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("PitchLattice dimensions must be positive")
        if self.frame_size % self.subframes:
            raise ValueError("frame size must be divisible by the subframe count")
        if self.feature_dim != self.spectral_features + 4:
            raise ValueError("the control contract is spectral features plus four controls")
        if self.max_period >= self.pitch_embedding_bins:
            raise ValueError("pitch embedding table does not cover the maximum period")
        if self.memory_size < 2 * self.max_period + self.subframe_size:
            raise ValueError("memory must cover two pitch periods and one subframe")
        if self.min_period <= 2:
            raise ValueError("minimum pitch period must leave room for lattice taps")
        if not math.isfinite(self.quantization_noise) or self.quantization_noise < 0.0:
            raise ValueError("quantization noise must be finite and non-negative")


def _orthogonal_recurrent(module: nn.Module) -> None:
    if isinstance(module, nn.GRUCell):
        nn.init.orthogonal_(module.weight_hh)


class GatedLinearUnit(nn.Module):
    """Multiplicative gate used after each compact recurrent projection."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.gate = nn.Linear(width, width, bias=False)
        nn.init.orthogonal_(self.gate.weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * torch.sigmoid(self.gate(value))


class FramewiseConv(nn.Module):
    """A two-step fully connected convolution with a recurrent input state."""

    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.projection = nn.Linear(2 * input_size, output_size, bias=False)
        self.gate = GatedLinearUnit(output_size)
        nn.init.orthogonal_(self.projection.weight)

    def forward(
        self,
        value: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if value.shape != state.shape:
            raise ValueError("FramewiseConv value and state shapes differ")
        joined = torch.cat([state, value], dim=-1)
        result = self.gate(torch.tanh(self.projection(joined)))
        return result, value


class PitchLatticeConditioner(nn.Module):
    """Turn frame controls and discrete pitch periods into four subframe codes."""

    def __init__(self, config: PitchLatticeConfig) -> None:
        super().__init__()
        self.config = config
        self.pitch_embedding = nn.Embedding(
            config.pitch_embedding_bins,
            config.pitch_embedding_dim,
        )
        self.input_dense = nn.Linear(
            config.feature_dim + config.pitch_embedding_dim,
            config.condition_hidden,
            bias=False,
        )
        self.context = nn.Conv1d(
            config.condition_hidden,
            config.condition_conv,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.output_dense = nn.Linear(
            config.condition_conv,
            config.subframes * config.subframe_condition,
            bias=False,
        )

    def forward(self, controls: torch.Tensor, periods: torch.Tensor) -> torch.Tensor:
        if controls.ndim != 3:
            raise ValueError("controls must have [batch, channels, frames] shape")
        if periods.ndim != 2:
            raise ValueError("periods must have [batch, frames] shape")
        if (
            controls.shape[0] != periods.shape[0]
            or controls.shape[2] != periods.shape[1]
        ):
            raise ValueError("control and period frame shapes differ")
        if controls.shape[1] != self.config.feature_dim:
            raise ValueError("unexpected PitchLattice control width")
        clipped = periods.to(dtype=torch.long).clamp(
            self.config.min_period,
            self.config.max_period,
        )
        value = torch.cat(
            [controls.transpose(1, 2), self.pitch_embedding(clipped)],
            dim=-1,
        )
        value = torch.tanh(self.input_dense(value)).transpose(1, 2)
        value = torch.tanh(self.context(value)).transpose(1, 2)
        value = torch.tanh(self.output_dense(value))
        return value.reshape(
            controls.shape[0],
            controls.shape[2],
            self.config.subframes,
            self.config.subframe_condition,
        )


class PitchLatticeSubframe(nn.Module):
    """Generate one waveform subframe using short- and long-term memory."""

    def __init__(self, config: PitchLatticeConfig) -> None:
        super().__init__()
        self.config = config
        size = config.subframe_size
        recurrent_input = 2 * size
        fwc_input = recurrent_input + config.subframe_condition + 4

        self.condition_gain = nn.Linear(config.subframe_condition, 1)
        self.lattice_mix = nn.Linear(config.subframe_condition, 2)
        self.fwc = FramewiseConv(fwc_input, config.fwc_width)
        self.gru1 = nn.GRUCell(
            config.fwc_width + recurrent_input,
            config.gru1_width,
            bias=False,
        )
        self.gru2 = nn.GRUCell(
            config.gru1_width + recurrent_input,
            config.gru2_width,
            bias=False,
        )
        self.gru3 = nn.GRUCell(
            config.gru2_width + recurrent_input,
            config.gru3_width,
            bias=False,
        )
        self.gru1_gate = GatedLinearUnit(config.gru1_width)
        self.gru2_gate = GatedLinearUnit(config.gru2_width)
        self.gru3_gate = GatedLinearUnit(config.gru3_width)
        skip_input = (
            config.fwc_width
            + config.gru1_width
            + config.gru2_width
            + config.gru3_width
            + recurrent_input
        )
        self.skip_dense = nn.Linear(skip_input, config.skip_width, bias=False)
        self.skip_gate = GatedLinearUnit(config.skip_width)
        self.waveform = nn.Linear(config.skip_width, size, bias=False)
        self.pitch_gains = nn.Linear(config.fwc_width, 4)

        self.apply(_orthogonal_recurrent)
        nn.init.constant_(self.condition_gain.bias, math.log(0.03))

    def _qnoise(self, value: torch.Tensor) -> torch.Tensor:
        amount = self.config.quantization_noise
        if not self.training or amount == 0.0:
            return value
        return (value + amount * (torch.rand_like(value) - 0.5)).clamp(-1.0, 1.0)

    def _pitch_taps(
        self,
        memory: torch.Tensor,
        periods: torch.Tensor,
        multiplier: int,
    ) -> torch.Tensor:
        size = self.config.subframe_size
        offsets = torch.arange(size + 4, device=memory.device, dtype=torch.long) - 2
        lag = periods[:, None] * multiplier
        indices = self.config.memory_size - lag + offsets[None]
        # A pitch shorter than one subframe crosses the newest memory boundary.
        # Folding by one period preserves a valid causal sample at that phase.
        indices = torch.where(
            indices >= self.config.memory_size,
            indices - periods[:, None],
            indices,
        )
        if bool(torch.any(indices < 0)) or bool(torch.any(indices >= self.config.memory_size)):
            raise RuntimeError("pitch lattice index escaped excitation memory")
        return torch.gather(memory, 1, indices)

    def forward(
        self,
        condition: torch.Tensor,
        memory: torch.Tensor,
        periods: torch.Tensor,
        voicing: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        gain = torch.exp(self.condition_gain(condition).clamp(-7.0, 1.0))
        one_period = self._pitch_taps(memory, periods, 1)
        two_periods = self._pitch_taps(memory, periods, 2)
        mixture = torch.softmax(self.lattice_mix(condition), dim=-1)
        prediction = (
            mixture[:, 0:1] * one_period + mixture[:, 1:2] * two_periods
        )
        prediction = self._qnoise(prediction / gain.clamp_min(1e-5))
        pitch = prediction[:, 2:-2] * voicing[:, None]
        previous = self._qnoise(
            memory[:, -self.config.subframe_size :] / gain.clamp_min(1e-5)
        )

        combined = torch.cat([condition, prediction, previous], dim=-1)
        fwc, fwc_state = self.fwc(combined, states[3])
        fwc = self._qnoise(fwc)
        pitch_gains = torch.sigmoid(self.pitch_gains(fwc))

        gru1_state = self.gru1(
            torch.cat([fwc, pitch_gains[:, 0:1] * pitch, previous], dim=-1),
            states[0],
        )
        gru1 = self._qnoise(self.gru1_gate(self._qnoise(gru1_state)))
        gru2_state = self.gru2(
            torch.cat([gru1, pitch_gains[:, 1:2] * pitch, previous], dim=-1),
            states[1],
        )
        gru2 = self._qnoise(self.gru2_gate(self._qnoise(gru2_state)))
        gru3_state = self.gru3(
            torch.cat([gru2, pitch_gains[:, 2:3] * pitch, previous], dim=-1),
            states[2],
        )
        gru3 = self._qnoise(self.gru3_gate(self._qnoise(gru3_state)))
        skip = torch.cat(
            [
                fwc,
                gru1,
                gru2,
                gru3,
                pitch_gains[:, 3:4] * pitch,
                previous,
            ],
            dim=-1,
        )
        skip = self.skip_gate(self._qnoise(torch.tanh(self.skip_dense(skip))))
        output = torch.tanh(self.waveform(skip)) * gain
        memory = torch.cat([memory[:, self.config.subframe_size :], output], dim=-1)
        return output, memory, (gru1_state, gru2_state, gru3_state, fwc_state)


class PitchLatticeVocoder(nn.Module):
    """Framewise waveform decoder with a dual-period excitation lattice."""

    def __init__(self, config: PitchLatticeConfig = PitchLatticeConfig()) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.conditioner = PitchLatticeConditioner(config)
        self.generator = PitchLatticeSubframe(config)

    def initial_states(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        config = self.config
        fwc_input = 2 * config.subframe_size + config.subframe_condition + 4
        return (
            torch.zeros(batch_size, config.gru1_width, device=device, dtype=dtype),
            torch.zeros(batch_size, config.gru2_width, device=device, dtype=dtype),
            torch.zeros(batch_size, config.gru3_width, device=device, dtype=dtype),
            torch.zeros(batch_size, fwc_input, device=device, dtype=dtype),
        )

    def forward(
        self,
        controls: torch.Tensor,
        periods: torch.Tensor,
        *,
        prefix: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if controls.ndim != 3 or controls.shape[1] != self.config.feature_dim:
            raise ValueError("controls must have [batch, feature_dim, frames] shape")
        if periods.shape != (controls.shape[0], controls.shape[2]):
            raise ValueError("period shape does not match control frames")
        batch, _, frames = controls.shape
        if frames <= 0:
            raise ValueError("PitchLattice cannot synthesize an empty control sequence")
        total_samples = frames * self.config.frame_size
        if prefix is not None:
            if prefix.ndim != 2 or prefix.shape[0] != batch:
                raise ValueError("prefix must have [batch, samples] shape")
            if prefix.shape[1] > total_samples:
                raise ValueError("prefix is longer than the requested waveform")
            if prefix.shape[1] % self.config.subframe_size:
                raise ValueError("prefix length must be subframe-aligned")

        conditions = self.conditioner(controls, periods)
        memory = torch.zeros(
            batch,
            self.config.memory_size,
            device=controls.device,
            dtype=controls.dtype,
        )
        states = self.initial_states(batch, device=controls.device, dtype=controls.dtype)
        outputs: list[torch.Tensor] = []
        prefix_samples = 0 if prefix is None else int(prefix.shape[1])
        for frame in range(frames):
            frame_period = periods[:, frame].to(dtype=torch.long).clamp(
                self.config.min_period,
                self.config.max_period,
            )
            voicing = controls[
                :, self.config.spectral_features + 1, frame
            ].clamp(0.0, 1.0)
            for subframe in range(self.config.subframes):
                output, memory, states = self.generator(
                    conditions[:, frame, subframe],
                    memory,
                    frame_period,
                    voicing,
                    states,
                )
                position = (frame * self.config.subframes + subframe) * self.config.subframe_size
                if position < prefix_samples:
                    if prefix is None:
                        raise RuntimeError("internal prefix accounting failed")
                    output = prefix[:, position : position + self.config.subframe_size]
                    memory = torch.cat(
                        [memory[:, : -self.config.subframe_size], output],
                        dim=-1,
                    )
                outputs.append(output)
        waveform = torch.cat(outputs, dim=-1)
        if waveform.shape != (batch, total_samples):
            raise RuntimeError("PitchLattice produced the wrong waveform length")
        return waveform


def pitch_lattice_parameter_count(model: PitchLatticeVocoder) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


__all__ = [
    "PitchLatticeConfig",
    "PitchLatticeVocoder",
    "pitch_lattice_parameter_count",
]
