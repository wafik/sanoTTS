"""SharedTensor Vocos: layer-mode tensor factorization of a wide decoder."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from saanotts.models.widebasis_vocoder import FactorizedDictionaryHead


@dataclass(frozen=True)
class SharedTensorVocoderConfig:
    mel_channels: int = 100
    prosody_channels: int = 3
    width: int = 512
    depth: int = 8
    expansion: int = 3
    embed_rank: int = 32
    shared_rank: int = 124
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
        values = tuple(self.to_dict().values())
        if any(not isinstance(value, int) or value <= 0 for value in values):
            raise ValueError("SharedTensor dimensions must be positive integers")
        if self.n_fft % 2 or self.hop_length > self.win_length:
            raise ValueError("invalid SharedTensor Fourier configuration")


class SharedTensorCore(nn.Module):
    """Input/output factors shared by every residual layer."""

    def __init__(self, config: SharedTensorVocoderConfig) -> None:
        super().__init__()
        intermediate = config.width * config.expansion
        rank = config.shared_rank
        self.up_input = nn.Linear(config.width, rank, bias=False)
        self.up_output = nn.Linear(rank, intermediate, bias=False)
        self.down_input = nn.Linear(intermediate, rank, bias=False)
        self.down_output = nn.Linear(rank, config.width, bias=False)


class SharedTensorBlock(nn.Module):
    """A layer-specific temporal filter and diagonal slice of shared factors."""

    def __init__(self, config: SharedTensorVocoderConfig) -> None:
        super().__init__()
        width = config.width
        intermediate = width * config.expansion
        self.depthwise = nn.Conv1d(
            width,
            width,
            kernel_size=7,
            padding=3,
            groups=width,
        )
        self.norm = nn.LayerNorm(width, eps=1e-6)
        self.up_gate = nn.Parameter(torch.ones(config.shared_rank))
        self.up_bias = nn.Parameter(torch.zeros(intermediate))
        self.down_gate = nn.Parameter(torch.ones(config.shared_rank))
        self.down_bias = nn.Parameter(torch.zeros(width))
        self.scale = nn.Parameter(torch.full((width,), 1.0 / config.depth))

    def forward(self, value: torch.Tensor, core: SharedTensorCore) -> torch.Tensor:
        residual = value
        value = self.depthwise(value).transpose(1, 2)
        value = self.norm(value)
        value = core.up_input(value) * self.up_gate
        value = F.linear(value, core.up_output.weight, self.up_bias)
        value = F.gelu(value)
        value = core.down_input(value) * self.down_gate
        value = F.linear(value, core.down_output.weight, self.down_bias)
        value = value * self.scale
        return residual + value.transpose(1, 2)


class SharedTensorVocoder(nn.Module):
    """Public-Vocos width/depth with CP factors shared across the layer mode."""

    def __init__(
        self,
        config: SharedTensorVocoderConfig = SharedTensorVocoderConfig(),
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
        self.core = SharedTensorCore(config)
        self.blocks = nn.ModuleList(
            [SharedTensorBlock(config) for _ in range(config.depth)]
        )
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

    def encode(
        self,
        mel: torch.Tensor,
        prosody: torch.Tensor,
        *,
        return_layers: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        if mel.ndim != 3 or mel.shape[1] != self.config.mel_channels:
            raise ValueError("mel must have [batch, mel_channels, frames] shape")
        expected = (mel.shape[0], self.config.prosody_channels, mel.shape[2])
        if prosody.shape != expected:
            raise ValueError("prosody shape does not match mel frames")
        value = torch.cat([mel, prosody], dim=1)
        value = self.embed_expand(self.embed_temporal(value))
        value = self.input_norm(value.transpose(1, 2)).transpose(1, 2)
        layers: list[torch.Tensor] = []
        for block in self.blocks:
            value = block(value, self.core)
            if return_layers:
                layers.append(value)
        value = self.final_norm(value.transpose(1, 2))
        if return_layers:
            return value, layers
        return value

    def forward(self, mel: torch.Tensor, prosody: torch.Tensor) -> torch.Tensor:
        encoded = self.encode(mel, prosody)
        if isinstance(encoded, tuple):
            raise RuntimeError("SharedTensor encoder returned unexpected layer features")
        value = encoded
        log_magnitude = self.magnitude(value).transpose(1, 2)
        phase = self.phase(value).transpose(1, 2)
        magnitude = torch.exp(log_magnitude).clamp(max=100.0)
        return magnitude * torch.complex(torch.cos(phase), torch.sin(phase))

    def synthesize(self, spectrum: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
        if spectrum.ndim != 3 or spectrum.shape[1] != self.config.bins:
            raise ValueError("spectrum must have [batch, bins, frames] shape")
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


def sharedtensor_parameter_breakdown(model: SharedTensorVocoder) -> dict[str, int]:
    values = {
        "embed": sum(parameter.numel() for parameter in model.embed_temporal.parameters())
        + sum(parameter.numel() for parameter in model.embed_expand.parameters()),
        "input_norm": sum(parameter.numel() for parameter in model.input_norm.parameters()),
        "shared_core": sum(parameter.numel() for parameter in model.core.parameters()),
        "layer_slices": sum(parameter.numel() for parameter in model.blocks.parameters()),
        "final_norm": sum(parameter.numel() for parameter in model.final_norm.parameters()),
        "magnitude_head": sum(parameter.numel() for parameter in model.magnitude.parameters()),
        "phase_head": sum(parameter.numel() for parameter in model.phase.parameters()),
    }
    values["total"] = int(sum(values.values()))
    return {name: int(value) for name, value in values.items()}


__all__ = [
    "SharedTensorBlock",
    "SharedTensorCore",
    "SharedTensorVocoder",
    "SharedTensorVocoderConfig",
    "sharedtensor_parameter_breakdown",
]
