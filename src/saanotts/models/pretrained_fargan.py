"""Trainable sub-700k FARGAN initialized from the official Opus decoder.

The decoder keeps the official 16 kHz recurrent widths and explicit pitch
memory.  Five large matrices are represented by independent low-rank factors;
this removes parameters from projections rather than shrinking the recurrent
state that carries waveform phase.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class PretrainedFarganConfig:
    sample_rate: int = 16_000
    frame_size: int = 160
    subframes: int = 4
    feature_dim: int = 20
    pitch_feature_index: int = 18
    pitch_embedding_bins: int = 224
    pitch_embedding_dim: int = 12
    condition_hidden: int = 64
    condition_conv: int = 128
    condition_size: int = 80
    fwc_width: int = 192
    gru1_width: int = 160
    gru2_width: int = 128
    gru3_width: int = 128
    skip_width: int = 128
    pitch_memory: int = 256
    deemphasis: float = 0.85
    gru1_input_rank: int | None = 120
    gru1_recurrent_rank: int | None = 96
    gru2_input_rank: int | None = 96
    gru3_input_rank: int | None = 96
    skip_rank: int | None = 96

    @property
    def subframe_size(self) -> int:
        return self.frame_size // self.subframes

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def full_rank(self) -> PretrainedFarganConfig:
        return replace(
            self,
            gru1_input_rank=None,
            gru1_recurrent_rank=None,
            gru2_input_rank=None,
            gru3_input_rank=None,
            skip_rank=None,
        )

    def validate(self) -> None:
        dimensions = (
            self.sample_rate,
            self.frame_size,
            self.subframes,
            self.feature_dim,
            self.pitch_embedding_bins,
            self.pitch_embedding_dim,
            self.condition_hidden,
            self.condition_conv,
            self.condition_size,
            self.fwc_width,
            self.gru1_width,
            self.gru2_width,
            self.gru3_width,
            self.skip_width,
            self.pitch_memory,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("FARGAN dimensions must be positive")
        if self.frame_size % self.subframes:
            raise ValueError("frame size must be divisible by the subframe count")
        if self.pitch_feature_index >= self.feature_dim:
            raise ValueError("pitch feature index is outside the feature vector")
        if not 0.0 <= self.deemphasis < 1.0:
            raise ValueError("deemphasis must be in [0, 1)")
        for rank in (
            self.gru1_input_rank,
            self.gru1_recurrent_rank,
            self.gru2_input_rank,
            self.gru3_input_rank,
            self.skip_rank,
        ):
            if rank is not None and rank <= 0:
                raise ValueError("factor ranks must be positive or None")


def quantize_activation(value: torch.Tensor) -> torch.Tensor:
    """Match Opus' signed 8-bit activation grid with a straight-through gradient."""

    quantized = torch.floor(127.0 * value + 0.5) / 127.0
    if torch.is_grad_enabled() and value.requires_grad:
        return value + (quantized - value).detach()
    return quantized


class QuantizedLinear(nn.Module):
    """Dense projection whose input follows the Opus int8 activation contract."""

    def __init__(self, inputs: int, outputs: int) -> None:
        super().__init__()
        self.inputs = int(inputs)
        self.outputs = int(outputs)
        self.weight = nn.Parameter(torch.empty(outputs, inputs))
        nn.init.orthogonal_(self.weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.linear(quantize_activation(value), self.weight)


class FactorizedQuantizedLinear(nn.Module):
    """Low-rank replacement for one official quantized matrix."""

    def __init__(self, inputs: int, outputs: int, rank: int) -> None:
        super().__init__()
        self.inputs = int(inputs)
        self.outputs = int(outputs)
        self.rank = int(rank)
        self.input = nn.Linear(inputs, rank, bias=False)
        self.output = nn.Linear(rank, outputs, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.output(self.input(quantize_activation(value)))


def make_quantized_linear(
    inputs: int,
    outputs: int,
    rank: int | None = None,
) -> QuantizedLinear | FactorizedQuantizedLinear:
    if rank is None:
        return QuantizedLinear(inputs, outputs)
    if rank > min(inputs, outputs):
        raise ValueError(f"rank {rank} exceeds matrix shape {outputs}x{inputs}")
    return FactorizedQuantizedLinear(inputs, outputs, rank)


class OfficialGruCell(nn.Module):
    """The z/r/h GRU equation used by Opus rather than PyTorch's gate order."""

    def __init__(
        self,
        inputs: int,
        hidden: int,
        input_rank: int | None = None,
        recurrent_rank: int | None = None,
    ) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.input = make_quantized_linear(inputs, 3 * hidden, input_rank)
        self.recurrent = make_quantized_linear(hidden, 3 * hidden, recurrent_rank)

    def forward(self, value: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        projected = self.input(value)
        recurrent = self.recurrent(state)
        z, r, candidate = projected.split(self.hidden, dim=-1)
        recurrent_z, recurrent_r, recurrent_candidate = recurrent.split(
            self.hidden,
            dim=-1,
        )
        z = torch.sigmoid(z + recurrent_z)
        r = torch.sigmoid(r + recurrent_r)
        candidate = torch.tanh(candidate + r * recurrent_candidate)
        return z * state + (1.0 - z) * candidate


class OfficialGlu(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.gate = QuantizedLinear(width, width)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * torch.sigmoid(self.gate(value))


class PretrainedFargan(nn.Module):
    """Faithful official FARGAN signal path with selected low-rank matrices."""

    def __init__(
        self,
        config: PretrainedFarganConfig = PretrainedFarganConfig(),
    ) -> None:
        super().__init__()
        config.validate()
        self.config = config
        size = config.subframe_size

        self.pitch_embedding = nn.Embedding(
            config.pitch_embedding_bins,
            config.pitch_embedding_dim,
        )
        self.condition_input = nn.Linear(
            config.feature_dim + config.pitch_embedding_dim,
            config.condition_hidden,
            bias=False,
        )
        self.condition_context = QuantizedLinear(
            3 * config.condition_hidden,
            config.condition_conv,
        )
        self.condition_output = QuantizedLinear(
            config.condition_conv,
            config.subframes * config.condition_size,
        )

        fwc_input = config.condition_size + 2 * size + 4
        self.condition_gain = nn.Linear(config.condition_size, 1)
        self.fwc = QuantizedLinear(2 * fwc_input, config.fwc_width)
        self.fwc_gate = OfficialGlu(config.fwc_width)
        self.pitch_gains = nn.Linear(config.fwc_width, 4)
        self.gru1 = OfficialGruCell(
            config.fwc_width + 2 * size,
            config.gru1_width,
            config.gru1_input_rank,
            config.gru1_recurrent_rank,
        )
        self.gru2 = OfficialGruCell(
            config.gru1_width + 2 * size,
            config.gru2_width,
            config.gru2_input_rank,
        )
        self.gru3 = OfficialGruCell(
            config.gru2_width + 2 * size,
            config.gru3_width,
            config.gru3_input_rank,
        )
        self.gru1_gate = OfficialGlu(config.gru1_width)
        self.gru2_gate = OfficialGlu(config.gru2_width)
        self.gru3_gate = OfficialGlu(config.gru3_width)
        skip_inputs = (
            config.gru1_width
            + config.gru2_width
            + config.gru3_width
            + config.fwc_width
            + 2 * size
        )
        self.skip = make_quantized_linear(skip_inputs, config.skip_width, config.skip_rank)
        self.skip_gate = OfficialGlu(config.skip_width)
        self.waveform = QuantizedLinear(config.skip_width, size)

    def _period(self, features: torch.Tensor) -> torch.Tensor:
        pitch = features[..., self.config.pitch_feature_index]
        period = torch.floor(0.5 + 256.0 / torch.pow(2.0, pitch + 1.5))
        return period.to(dtype=torch.long).clamp(1, self.config.pitch_memory - 1)

    def _condition_frame(
        self,
        features: torch.Tensor,
        period: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        index = (period - 32).clamp(0, self.config.pitch_embedding_bins - 1)
        value = torch.cat([features, self.pitch_embedding(index)], dim=-1)
        value = torch.tanh(self.condition_input(value))
        joined = torch.cat([state, value], dim=-1)
        value = torch.tanh(self.condition_context(joined))
        value = torch.tanh(self.condition_output(value))
        return value.reshape(features.shape[0], self.config.subframes, -1), joined[:, 64:]

    def _pitch_prediction(
        self,
        memory: torch.Tensor,
        period: torch.Tensor,
    ) -> torch.Tensor:
        width = self.config.subframe_size + 4
        offsets = torch.arange(width, device=memory.device, dtype=torch.long) - 2
        indices = self.config.pitch_memory - period[:, None] + offsets[None]
        indices = torch.where(
            indices >= self.config.pitch_memory,
            indices - period[:, None],
            indices,
        ).clamp_min(0)
        return torch.gather(memory, 1, indices)

    def _run_subframe(
        self,
        condition: torch.Tensor,
        period: torch.Tensor,
        memory: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        deemphasis_memory: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        torch.Tensor,
    ]:
        gru1_state, gru2_state, gru3_state, fwc_state = states
        size = self.config.subframe_size
        gain = torch.exp(self.condition_gain(condition))
        prediction = self._pitch_prediction(memory, period)
        prediction = (prediction / (1e-5 + gain)).clamp(-1.0, 1.0)
        previous = (memory[:, -size:] / (1e-5 + gain)).clamp(-1.0, 1.0)
        pitch = prediction[:, 2:-2]

        fwc_input = torch.cat([condition, prediction, previous], dim=-1)
        fwc_joined = torch.cat([fwc_state, fwc_input], dim=-1)
        fwc_output = torch.tanh(self.fwc(fwc_joined))
        fwc_output = self.fwc_gate(fwc_output)
        pitch_gains = torch.sigmoid(self.pitch_gains(fwc_output))

        gru1_state = self.gru1(
            torch.cat([fwc_output, pitch_gains[:, 0:1] * pitch, previous], dim=-1),
            gru1_state,
        )
        gru1_output = self.gru1_gate(gru1_state)
        gru2_state = self.gru2(
            torch.cat([gru1_output, pitch_gains[:, 1:2] * pitch, previous], dim=-1),
            gru2_state,
        )
        gru2_output = self.gru2_gate(gru2_state)
        gru3_state = self.gru3(
            torch.cat([gru2_output, pitch_gains[:, 2:3] * pitch, previous], dim=-1),
            gru3_state,
        )
        gru3_output = self.gru3_gate(gru3_state)
        skip_input = torch.cat(
            [
                gru1_output,
                gru2_output,
                gru3_output,
                fwc_output,
                pitch_gains[:, 3:4] * pitch,
                previous,
            ],
            dim=-1,
        )
        skip = self.skip_gate(torch.tanh(self.skip(skip_input)))
        excitation = torch.tanh(self.waveform(skip)) * gain
        memory = torch.cat([memory[:, size:], excitation], dim=-1)

        output_samples = []
        value = deemphasis_memory
        for sample in excitation.unbind(dim=-1):
            value = sample + self.config.deemphasis * value
            output_samples.append(value)
        output = torch.stack(output_samples, dim=-1)
        return (
            output,
            memory,
            (gru1_state, gru2_state, gru3_state, fwc_input),
            value,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Synthesize `[batch, frames, >=20]` LPCNet features after five-frame warmup."""

        if features.ndim != 3 or features.shape[-1] < self.config.feature_dim:
            raise ValueError("features must have [batch, frames, at least 20] shape")
        if features.shape[1] < 6:
            raise ValueError("at least six feature frames are required")
        features = features[..., : self.config.feature_dim]
        batch = features.shape[0]
        options = {"device": features.device, "dtype": features.dtype}
        condition_state = torch.zeros(batch, 2 * self.config.condition_hidden, **options)
        memory = torch.zeros(batch, self.config.pitch_memory, **options)
        states = (
            torch.zeros(batch, self.config.gru1_width, **options),
            torch.zeros(batch, self.config.gru2_width, **options),
            torch.zeros(batch, self.config.gru3_width, **options),
            torch.zeros(
                batch,
                self.config.condition_size + 2 * self.config.subframe_size + 4,
                **options,
            ),
        )
        deemphasis_memory = torch.zeros(batch, **options)

        period = torch.zeros(batch, device=features.device, dtype=torch.long)
        condition = None
        last_period = period
        for frame in range(5):
            last_period = period
            period = self._period(features[:, frame])
            condition, condition_state = self._condition_frame(
                features[:, frame],
                period,
                condition_state,
            )
        if condition is None:
            raise RuntimeError("FARGAN continuation condition was not initialized")
        for subframe in range(self.config.subframes):
            _, memory, states, deemphasis_memory = self._run_subframe(
                condition[:, subframe],
                last_period,
                memory,
                states,
                deemphasis_memory,
            )
            memory = torch.cat(
                [
                    memory[:, : -self.config.subframe_size],
                    torch.zeros(batch, self.config.subframe_size, **options),
                ],
                dim=-1,
            )
        deemphasis_memory = torch.zeros_like(deemphasis_memory)

        output = []
        for frame in range(5, features.shape[1]):
            period = self._period(features[:, frame])
            condition, condition_state = self._condition_frame(
                features[:, frame],
                period,
                condition_state,
            )
            for subframe in range(self.config.subframes):
                sub_output, memory, states, deemphasis_memory = self._run_subframe(
                    condition[:, subframe],
                    last_period,
                    memory,
                    states,
                    deemphasis_memory,
                )
                output.append(sub_output)
            last_period = period
        return torch.cat(output, dim=-1)


def pretrained_fargan_parameter_breakdown(model: PretrainedFargan) -> dict[str, int]:
    groups = {
        "conditioner": (
            model.pitch_embedding,
            model.condition_input,
            model.condition_context,
            model.condition_output,
        ),
        "gain_and_fwc": (
            model.condition_gain,
            model.fwc,
            model.fwc_gate,
            model.pitch_gains,
        ),
        "recurrent": (
            model.gru1,
            model.gru2,
            model.gru3,
            model.gru1_gate,
            model.gru2_gate,
            model.gru3_gate,
        ),
        "output": (model.skip, model.skip_gate, model.waveform),
    }
    result = {
        name: sum(parameter.numel() for module in modules for parameter in module.parameters())
        for name, modules in groups.items()
    }
    result["total"] = sum(result.values())
    return {name: int(value) for name, value in result.items()}


__all__ = [
    "FactorizedQuantizedLinear",
    "OfficialGruCell",
    "PretrainedFargan",
    "PretrainedFarganConfig",
    "QuantizedLinear",
    "make_quantized_linear",
    "pretrained_fargan_parameter_breakdown",
    "quantize_activation",
]
