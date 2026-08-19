"""WideBasis Vocos: preserve teacher width and depth under a sub-700k budget."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class WideBasisVocoderConfig:
    mel_channels: int = 100
    prosody_channels: int = 3
    width: int = 512
    depth: int = 8
    expansion: int = 3
    embed_rank: int = 48
    block_rank: int = 15
    magnitude_rank: int = 32
    phase_rank: int = 40
    n_fft: int = 1_024
    hop_length: int = 256
    win_length: int = 1_024
    sample_rate: int = 24_000

    @property
    def bins(self) -> int:
        return self.n_fft // 2 + 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        dimensions = (
            self.mel_channels,
            self.prosody_channels,
            self.width,
            self.depth,
            self.expansion,
            self.embed_rank,
            self.block_rank,
            self.magnitude_rank,
            self.phase_rank,
            self.n_fft,
            self.hop_length,
            self.win_length,
            self.sample_rate,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("WideBasis dimensions must be positive")
        if self.n_fft % 2 or self.hop_length > self.win_length:
            raise ValueError("invalid WideBasis Fourier configuration")


class WideBasisBlock(nn.Module):
    """Full-width ConvNeXt residual block with rank-limited channel updates."""

    def __init__(self, config: WideBasisVocoderConfig) -> None:
        super().__init__()
        width = config.width
        intermediate = width * config.expansion
        rank = config.block_rank
        self.depthwise = nn.Conv1d(
            width,
            width,
            kernel_size=7,
            padding=3,
            groups=width,
        )
        self.norm = nn.LayerNorm(width, eps=1e-6)
        self.up_in = nn.Linear(width, rank, bias=False)
        self.up_out = nn.Linear(rank, intermediate)
        self.activation = nn.GELU()
        self.down_in = nn.Linear(intermediate, rank, bias=False)
        self.down_out = nn.Linear(rank, width)
        self.scale = nn.Parameter(torch.full((width,), 1.0 / config.depth))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = value
        value = self.depthwise(value).transpose(1, 2)
        value = self.norm(value)
        value = self.up_out(self.up_in(value))
        value = self.activation(value)
        value = self.down_out(self.down_in(value))
        value = value * self.scale
        return residual + value.transpose(1, 2)


class FactorizedDictionaryHead(nn.Module):
    def __init__(self, input_width: int, rank: int, output_width: int) -> None:
        super().__init__()
        self.input = nn.Linear(input_width, rank, bias=False)
        self.output = nn.Linear(rank, output_width)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.output(self.input(value))


class WideBasisVocoder(nn.Module):
    """Full-width, full-depth spectral decoder with low-rank learned updates."""

    def __init__(
        self,
        config: WideBasisVocoderConfig = WideBasisVocoderConfig(),
    ) -> None:
        super().__init__()
        config.validate()
        self.config = config
        inputs = config.mel_channels + config.prosody_channels
        self.embed_temporal = nn.Conv1d(
            inputs,
            config.embed_rank,
            kernel_size=7,
            padding=3,
        )
        self.embed_expand = nn.Conv1d(config.embed_rank, config.width, kernel_size=1)
        self.input_norm = nn.LayerNorm(config.width, eps=1e-6)
        self.blocks = nn.ModuleList([WideBasisBlock(config) for _ in range(config.depth)])
        self.final_norm = nn.LayerNorm(config.width, eps=1e-6)
        self.magnitude = FactorizedDictionaryHead(
            config.width,
            config.magnitude_rank,
            config.bins,
        )
        self.phase = FactorizedDictionaryHead(
            config.width,
            config.phase_rank,
            config.bins,
        )

    def forward(self, mel: torch.Tensor, prosody: torch.Tensor) -> torch.Tensor:
        if mel.ndim != 3 or mel.shape[1] != self.config.mel_channels:
            raise ValueError("mel must have [batch, mel_channels, frames] shape")
        if prosody.shape != (
            mel.shape[0],
            self.config.prosody_channels,
            mel.shape[2],
        ):
            raise ValueError("prosody shape does not match mel frames")
        value = torch.cat([mel, prosody], dim=1)
        value = self.embed_expand(self.embed_temporal(value))
        value = self.input_norm(value.transpose(1, 2)).transpose(1, 2)
        for block in self.blocks:
            value = block(value)
        value = self.final_norm(value.transpose(1, 2))
        log_magnitude = self.magnitude(value).transpose(1, 2)
        phase = self.phase(value).transpose(1, 2)
        magnitude = torch.exp(log_magnitude).clamp(max=100.0)
        return magnitude * torch.complex(torch.cos(phase), torch.sin(phase))

    def synthesize(self, spectrum: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
        if spectrum.ndim != 3 or spectrum.shape[1] != self.config.bins:
            raise ValueError("spectrum must have [batch, bins, frames] shape")
        if window.shape != (self.config.win_length,):
            raise ValueError("unexpected synthesis window")
        return torch.istft(
            spectrum,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            window=window.to(device=spectrum.device, dtype=spectrum.real.dtype),
            center=True,
            normalized=False,
            onesided=True,
            length=(spectrum.shape[-1] - 1) * self.config.hop_length,
        )


def widebasis_parameter_breakdown(model: WideBasisVocoder) -> dict[str, int]:
    values = {
        "embed": sum(parameter.numel() for parameter in model.embed_temporal.parameters())
        + sum(parameter.numel() for parameter in model.embed_expand.parameters()),
        "input_norm": sum(parameter.numel() for parameter in model.input_norm.parameters()),
        "blocks": sum(parameter.numel() for parameter in model.blocks.parameters()),
        "final_norm": sum(parameter.numel() for parameter in model.final_norm.parameters()),
        "magnitude_head": sum(parameter.numel() for parameter in model.magnitude.parameters()),
        "phase_head": sum(parameter.numel() for parameter in model.phase.parameters()),
    }
    values["total"] = int(sum(values.values()))
    return {name: int(value) for name, value in values.items()}


__all__ = [
    "WideBasisBlock",
    "WideBasisVocoder",
    "WideBasisVocoderConfig",
    "widebasis_parameter_breakdown",
]
