"""Recurrent shared-depth output head for frozen acoustic representations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from saanotts.models.rift_tts import SharedLoopCell, count_parameters


@dataclass(frozen=True)
class RecurrentTransplantConfig:
    channels: int = 96
    out_channels: int = 100
    steps: int = 4
    shared_kernel_size: int = 7
    expansion: int = 2
    output_kernel_size: int = 5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RecurrentTransplantHead(nn.Module):
    """Refine frozen frame states with one reused cell and a temporal head."""

    def __init__(self, config: RecurrentTransplantConfig) -> None:
        super().__init__()
        if config.channels <= 0 or config.out_channels <= 0 or config.steps <= 0:
            raise ValueError("invalid recurrent transplant dimensions")
        if config.output_kernel_size <= 0 or config.output_kernel_size % 2 == 0:
            raise ValueError("output kernel size must be positive and odd")
        self.config = config
        self.shared_cell = SharedLoopCell(
            config.channels,
            config.shared_kernel_size,
            config.expansion,
        )
        self.loop_bias = nn.Parameter(torch.zeros(config.steps, config.channels))
        self.output = nn.Conv1d(
            config.channels,
            config.out_channels,
            config.output_kernel_size,
            padding=0,
        )

    def initialize_from_linear_radius_head(self, weights: torch.Tensor) -> None:
        radius = self.config.output_kernel_size // 2
        expected = (self.config.channels * (2 * radius + 1) + 1, self.config.out_channels)
        if tuple(weights.shape) != expected:
            raise ValueError(f"linear head shape {tuple(weights.shape)} != {expected}")
        kernels = weights[:-1].reshape(
            2 * radius + 1,
            self.config.channels,
            self.config.out_channels,
        )
        with torch.no_grad():
            self.output.weight.copy_(kernels.permute(2, 1, 0))
            self.output.bias.copy_(weights[-1])
            self.shared_cell.residual_scale.zero_()
            self.loop_bias.zero_()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3 or hidden.shape[1] != self.config.channels or hidden.shape[-1] <= 0:
            raise ValueError(
                f"expected hidden [B,{self.config.channels},T], got {tuple(hidden.shape)}"
            )
        value = hidden
        for step in range(self.config.steps):
            value = self.shared_cell(value, self.loop_bias[step][None, :, None])
        radius = self.config.output_kernel_size // 2
        return self.output(F.pad(value, (radius, radius), mode="replicate"))


def recurrent_transplant_parameter_breakdown(model: RecurrentTransplantHead) -> dict[str, int]:
    values = {
        "shared_recurrent_cell": count_parameters(model.shared_cell),
        "loop_bias": int(model.loop_bias.numel()),
        "temporal_output_head": count_parameters(model.output),
    }
    values["total"] = count_parameters(model)
    return values


__all__ = [
    "RecurrentTransplantConfig",
    "RecurrentTransplantHead",
    "recurrent_transplant_parameter_breakdown",
]
