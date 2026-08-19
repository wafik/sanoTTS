"""Trellis-RIFT: tensor-factorized text conditioning for a frozen decoder.

The model keeps independent token- and frame-depth residual blocks, but each
dense temporal convolution is represented as a channel projection, a
depthwise temporal filter, and a channel expansion.  This preserves the
qualified teacher's two-axis topology while spending parameters on a compact
CP-style channel/time/channel trellis rather than dense convolution kernels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from saanotts.models.recurrent_transplant import (
    RecurrentTransplantConfig,
    RecurrentTransplantHead,
)
from saanotts.models.rift_tts import count_parameters


@dataclass(frozen=True)
class TrellisRiftConfig:
    vocab_size: int = 62
    duration_width: int = 64
    duration_depth: int = 3
    duration_kernel_size: int = 5
    duration_rank: int = 24
    acoustic_width: int = 96
    token_depth: int = 3
    frame_depth: int = 4
    acoustic_kernel_size: int = 5
    acoustic_rank: int = 56
    n_mels: int = 100
    head_steps: int = 4
    head_kernel_size: int = 7
    head_expansion: int = 2
    output_kernel_size: int = 5
    max_tokens: int = 207
    min_duration: int = 1
    max_duration: int = 80

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FactorizedTemporalConv1d(nn.Module):
    """A CP-style channel/time/channel replacement for a dense 1-D kernel."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        rank: int,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0 or rank <= 0:
            raise ValueError("factorized convolution dimensions must be positive")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("factorized convolution kernel must be positive and odd")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.rank = int(rank)
        self.reduce = nn.Conv1d(in_channels, rank, 1)
        self.temporal = nn.Conv1d(
            rank,
            rank,
            kernel_size,
            padding=kernel_size // 2,
            groups=rank,
        )
        self.expand = nn.Conv1d(rank, out_channels, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.expand(self.temporal(self.reduce(value)))

    @torch.no_grad()
    def initialize_from_dense(self, source: nn.Conv1d, *, als_iterations: int = 25) -> float:
        """Nested-SVD initialization from a same-padded dense convolution.

        The outer SVD separates output channels from input/time.  A rank-one
        SVD of each right singular component then separates input channel from
        time, producing a deterministic CP-style approximation.  The returned
        value is the relative Frobenius reconstruction error of the kernel.
        """

        expected_padding = (self.kernel_size // 2,)
        if (
            source.in_channels != self.in_channels
            or source.out_channels != self.out_channels
            or source.kernel_size != (self.kernel_size,)
            or source.groups != 1
            or source.stride != (1,)
            or source.dilation != (1,)
            or source.padding != expected_padding
        ):
            raise ValueError("dense convolution is incompatible with factorized target")
        if als_iterations < 0:
            raise ValueError("ALS iteration count must be non-negative")
        weight = source.weight.detach().to(dtype=torch.float32)
        flat = weight.reshape(self.out_channels, self.in_channels * self.kernel_size)
        left, singular, right = torch.linalg.svd(flat, full_matrices=False)
        used = min(self.rank, int(singular.numel()))

        self.reduce.weight.zero_()
        self.reduce.bias.zero_()
        self.temporal.weight.zero_()
        self.temporal.bias.zero_()
        self.expand.weight.zero_()
        self.expand.bias.zero_()
        for index in range(used):
            component = right[index].reshape(self.in_channels, self.kernel_size)
            channel, inner_singular, temporal = torch.linalg.svd(
                component,
                full_matrices=False,
            )
            scale = torch.sqrt(inner_singular[0].clamp_min(0.0))
            self.reduce.weight[index, :, 0].copy_(channel[:, 0] * scale)
            self.temporal.weight[index, 0].copy_(temporal[0] * scale)
            self.expand.weight[:, index, 0].copy_(left[:, index] * singular[index])
        if als_iterations:
            target = weight.to(dtype=torch.float64)
            output_factor = self.expand.weight[:, :, 0].detach().to(dtype=torch.float64)
            input_factor = self.reduce.weight[:, :, 0].detach().T.to(dtype=torch.float64)
            time_factor = self.temporal.weight[:, 0].detach().T.to(dtype=torch.float64)
            for _ in range(als_iterations):
                design = torch.einsum("ir,kr->ikr", input_factor, time_factor).reshape(
                    self.in_channels * self.kernel_size,
                    self.rank,
                )
                output_factor = torch.linalg.lstsq(
                    design,
                    target.reshape(self.out_channels, -1).T,
                ).solution.T
                design = torch.einsum("or,kr->okr", output_factor, time_factor).reshape(
                    self.out_channels * self.kernel_size,
                    self.rank,
                )
                input_factor = torch.linalg.lstsq(
                    design,
                    target.permute(1, 0, 2).reshape(self.in_channels, -1).T,
                ).solution.T
                design = torch.einsum("or,ir->oir", output_factor, input_factor).reshape(
                    self.out_channels * self.in_channels,
                    self.rank,
                )
                time_factor = torch.linalg.lstsq(
                    design,
                    target.permute(2, 0, 1).reshape(self.kernel_size, -1).T,
                ).solution.T

            input_norm = torch.linalg.vector_norm(input_factor, dim=0).clamp_min(1e-12)
            time_norm = torch.linalg.vector_norm(time_factor, dim=0).clamp_min(1e-12)
            output_factor = output_factor * input_norm[None] * time_norm[None]
            input_factor = input_factor / input_norm[None]
            time_factor = time_factor / time_norm[None]
            self.expand.weight[:, :, 0].copy_(output_factor.to(dtype=weight.dtype))
            self.reduce.weight[:, :, 0].copy_(input_factor.T.to(dtype=weight.dtype))
            self.temporal.weight[:, 0].copy_(time_factor.T.to(dtype=weight.dtype))
        if source.bias is not None:
            self.expand.bias.copy_(source.bias)

        reconstructed = torch.einsum(
            "or,ri,rk->oik",
            self.expand.weight[:, :, 0],
            self.reduce.weight[:, :, 0],
            self.temporal.weight[:, 0],
        )
        denominator = torch.linalg.vector_norm(weight).clamp_min(1e-12)
        return float((torch.linalg.vector_norm(reconstructed - weight) / denominator).cpu())


class TrellisResidualBlock(nn.Module):
    """Teacher-shaped residual block with two factorized temporal kernels."""

    def __init__(self, channels: int, kernel_size: int, rank: int) -> None:
        super().__init__()
        self.first = FactorizedTemporalConv1d(channels, channels, kernel_size, rank)
        self.second = FactorizedTemporalConv1d(channels, channels, kernel_size, rank)
        self.scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, value: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        mask_f = None if mask is None else mask.unsqueeze(1).to(dtype=value.dtype)
        source = value if mask_f is None else value * mask_f
        result = value + self.scale * self.second(F.silu(self.first(source)))
        return result if mask_f is None else result * mask_f

    @torch.no_grad()
    def initialize_from_dense(self, source: nn.Module) -> dict[str, float]:
        if not hasattr(source, "net") or not hasattr(source, "scale"):
            raise ValueError("source is not a compatible residual block")
        first = source.net[0]
        second = source.net[2]
        if not isinstance(first, nn.Conv1d) or not isinstance(second, nn.Conv1d):
            raise ValueError("source residual block does not contain dense convolutions")
        self.scale.copy_(source.scale)
        return {
            "first": self.first.initialize_from_dense(first),
            "second": self.second.initialize_from_dense(second),
        }


class TrellisDurationPredictor(nn.Module):
    """Tensor-factorized duration path with the qualified teacher's topology."""

    def __init__(self, config: TrellisRiftConfig) -> None:
        super().__init__()
        width = config.duration_width
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, width)
        self.input_proj = nn.Conv1d(width + 3, width, 1)
        self.blocks = nn.ModuleList(
            [
                TrellisResidualBlock(
                    width,
                    config.duration_kernel_size,
                    config.duration_rank,
                )
                for _ in range(config.duration_depth)
            ]
        )
        self.output = nn.Conv1d(width, 1, 1)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if ids.ndim != 2 or mask.ndim != 2 or ids.shape != mask.shape:
            raise ValueError("duration ids and mask must have the same [B,N] shape")
        batch, tokens = ids.shape
        if tokens <= 0:
            raise ValueError("duration input is empty")
        positions = torch.linspace(0.0, 1.0, tokens, device=ids.device).expand(batch, tokens)
        lengths = mask.sum(dim=1).clamp_min(1).to(dtype=torch.float32)
        length_hint = (
            torch.log1p(lengths) / math.log1p(float(self.config.max_tokens))
        ).unsqueeze(1).expand(batch, tokens)
        features = torch.stack(
            [positions, length_hint, mask.to(dtype=torch.float32)],
            dim=1,
        )
        value = self.embedding(ids).transpose(1, 2)
        value = self.input_proj(torch.cat([value, features.to(value.dtype)], dim=1))
        for block in self.blocks:
            value = block(value, mask)
        return self.output(value).squeeze(1) * mask.to(dtype=value.dtype)

    @torch.no_grad()
    def durations_from_prediction(self, log_durations: torch.Tensor) -> torch.Tensor:
        predicted = torch.round(torch.exp(log_durations).clamp_min(1.0))
        return predicted.clamp(
            min=float(self.config.min_duration),
            max=float(self.config.max_duration),
        ).to(dtype=torch.long)

    @torch.no_grad()
    def initialize_from_teacher(self, source: nn.Module) -> list[dict[str, float]]:
        if len(source.blocks) != len(self.blocks):
            raise ValueError("duration teacher depth changed")
        self.embedding.load_state_dict(source.embedding.state_dict(), strict=True)
        self.input_proj.load_state_dict(source.input_proj.state_dict(), strict=True)
        self.output.load_state_dict(source.output.state_dict(), strict=True)
        return [
            target.initialize_from_dense(origin)
            for target, origin in zip(self.blocks, source.blocks, strict=True)
        ]


class TrellisAcousticPath(nn.Module):
    """Independent factorized token and frame paths producing teacher-width states."""

    def __init__(self, config: TrellisRiftConfig) -> None:
        super().__init__()
        width = config.acoustic_width
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, width)
        self.token_input_proj = nn.Conv1d(width + 2, width, 1)
        self.token_blocks = nn.ModuleList(
            [
                TrellisResidualBlock(
                    width,
                    config.acoustic_kernel_size,
                    config.acoustic_rank,
                )
                for _ in range(config.token_depth)
            ]
        )
        self.frame_input_proj = nn.Conv1d(width + 3, width, 1)
        self.frame_blocks = nn.ModuleList(
            [
                TrellisResidualBlock(
                    width,
                    config.acoustic_kernel_size,
                    config.acoustic_rank,
                )
                for _ in range(config.frame_depth)
            ]
        )

    def forward(self, ids: torch.Tensor, durations: torch.Tensor) -> torch.Tensor:
        if ids.ndim != 2 or durations.ndim != 2 or ids.shape != durations.shape:
            raise ValueError("acoustic ids and durations must have the same [B,N] shape")
        if ids.shape[0] != 1:
            raise ValueError("Trellis-RIFT acoustic inference currently requires batch size one")
        if bool(torch.any(durations <= 0)):
            raise ValueError("all acoustic durations must be positive")
        token_count = ids.shape[1]
        token_pos = torch.linspace(0.0, 1.0, token_count, device=ids.device)
        durations_f = durations[0].to(dtype=torch.float32)
        duration_hint = torch.log1p(durations_f) / torch.log1p(durations_f.max().clamp_min(1.0))
        token_features = torch.stack([token_pos, duration_hint], dim=0).unsqueeze(0)
        value = self.embedding(ids).transpose(1, 2)
        value = self.token_input_proj(torch.cat([value, token_features.to(value.dtype)], dim=1))
        for block in self.token_blocks:
            value = block(value)

        frames = int(durations.sum().item())
        contextual = value[0].transpose(0, 1)
        expanded = torch.repeat_interleave(contextual, durations[0], dim=0)
        if expanded.shape[0] != frames:
            raise RuntimeError("Trellis-RIFT length regulator produced the wrong frame count")
        frame_pos = torch.linspace(0.0, 1.0, frames, device=ids.device)
        token_position_parts: list[torch.Tensor] = []
        duration_position_parts: list[torch.Tensor] = []
        denominator = max(token_count - 1, 1)
        for index, duration_value in enumerate(durations[0].unbind()):
            duration = int(duration_value.item())
            token_position_parts.append(
                torch.full((duration,), float(index) / float(denominator), device=ids.device)
            )
            duration_position_parts.append(
                torch.linspace(0.0, 1.0, duration, device=ids.device)
            )
        frame_features = torch.stack(
            [
                frame_pos,
                torch.cat(token_position_parts),
                torch.cat(duration_position_parts),
            ],
            dim=0,
        ).unsqueeze(0)
        value = expanded.transpose(0, 1).unsqueeze(0)
        value = self.frame_input_proj(torch.cat([value, frame_features.to(value.dtype)], dim=1))
        for block in self.frame_blocks:
            value = block(value)
        return value

    @torch.no_grad()
    def initialize_from_teacher(self, source: nn.Module) -> dict[str, list[dict[str, float]]]:
        if len(source.token_blocks) != len(self.token_blocks):
            raise ValueError("acoustic teacher token depth changed")
        if len(source.frame_blocks) != len(self.frame_blocks):
            raise ValueError("acoustic teacher frame depth changed")
        self.embedding.load_state_dict(source.embedding.state_dict(), strict=True)
        self.token_input_proj.load_state_dict(source.token_input_proj.state_dict(), strict=True)
        self.frame_input_proj.load_state_dict(source.frame_input_proj.state_dict(), strict=True)
        return {
            "token": [
                target.initialize_from_dense(origin)
                for target, origin in zip(
                    self.token_blocks,
                    source.token_blocks,
                    strict=True,
                )
            ],
            "frame": [
                target.initialize_from_dense(origin)
                for target, origin in zip(
                    self.frame_blocks,
                    source.frame_blocks,
                    strict=True,
                )
            ],
        }


class TrellisRift(nn.Module):
    """Complete phoneme-to-decoder-coordinate Trellis-RIFT front end."""

    def __init__(self, config: TrellisRiftConfig) -> None:
        super().__init__()
        if config.vocab_size <= 0 or config.n_mels <= 0:
            raise ValueError("Trellis-RIFT vocabulary and mel dimensions must be positive")
        if config.acoustic_width != 96:
            raise ValueError("the qualified recurrent head requires acoustic width 96")
        self.config = config
        self.duration = TrellisDurationPredictor(config)
        self.acoustic = TrellisAcousticPath(config)
        self.head = RecurrentTransplantHead(
            RecurrentTransplantConfig(
                channels=config.acoustic_width,
                out_channels=config.n_mels,
                steps=config.head_steps,
                shared_kernel_size=config.head_kernel_size,
                expansion=config.head_expansion,
                output_kernel_size=config.output_kernel_size,
            )
        )

    def forward(
        self,
        ids: torch.Tensor,
        durations: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        mask = torch.ones_like(ids, dtype=torch.bool)
        log_durations = self.duration(ids, mask)
        used_durations = (
            self.duration.durations_from_prediction(log_durations)
            if durations is None
            else durations.to(device=ids.device, dtype=torch.long)
        )
        hidden = self.acoustic(ids, used_durations)
        return {
            "mel": self.head(hidden),
            "hidden": hidden,
            "log_durations": log_durations,
            "durations": used_durations,
        }

    @torch.no_grad()
    def initialize_from_teacher(
        self,
        duration: nn.Module,
        acoustic: nn.Module,
        head: RecurrentTransplantHead,
    ) -> dict[str, Any]:
        if head.config != self.head.config:
            raise ValueError("teacher recurrent-head configuration changed")
        errors = {
            "duration": self.duration.initialize_from_teacher(duration),
            "acoustic": self.acoustic.initialize_from_teacher(acoustic),
        }
        self.head.load_state_dict(head.state_dict(), strict=True)
        return errors


def trellis_rift_parameter_breakdown(model: TrellisRift) -> dict[str, int]:
    values = {
        "duration": count_parameters(model.duration),
        "acoustic": count_parameters(model.acoustic),
        "recurrent_head": count_parameters(model.head),
    }
    values["front_total"] = int(sum(values.values()))
    return values


__all__ = [
    "FactorizedTemporalConv1d",
    "TrellisAcousticPath",
    "TrellisDurationPredictor",
    "TrellisResidualBlock",
    "TrellisRift",
    "TrellisRiftConfig",
    "trellis_rift_parameter_breakdown",
]
