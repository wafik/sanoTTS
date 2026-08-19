"""Pitch-aware acoustic front for the PitchLattice waveform decoder."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from saanotts.models.trellis_rift import (
    FactorizedTemporalConv1d,
    TrellisAcousticPath,
    TrellisDurationPredictor,
    TrellisRift,
    TrellisRiftConfig,
)


@dataclass(frozen=True)
class PitchLatticeAcousticConfig:
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
    spectral_features: int = 20
    detail_kernel_size: int = 7
    detail_rank: int = 32
    control_kernel_size: int = 7
    control_rank: int = 12
    max_tokens: int = 207
    min_duration: int = 1
    max_duration: int = 80

    @property
    def feature_dim(self) -> int:
        return self.spectral_features + 4

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def trellis_config(self) -> TrellisRiftConfig:
        return TrellisRiftConfig(
            vocab_size=self.vocab_size,
            duration_width=self.duration_width,
            duration_depth=self.duration_depth,
            duration_kernel_size=self.duration_kernel_size,
            duration_rank=self.duration_rank,
            acoustic_width=self.acoustic_width,
            token_depth=self.token_depth,
            frame_depth=self.frame_depth,
            acoustic_kernel_size=self.acoustic_kernel_size,
            acoustic_rank=self.acoustic_rank,
            n_mels=self.feature_dim,
            max_tokens=self.max_tokens,
            min_duration=self.min_duration,
            max_duration=self.max_duration,
        )


class PitchLatticeAcoustic(nn.Module):
    """Predict spectral envelope, temporal detail, and excitation controls."""

    def __init__(
        self,
        config: PitchLatticeAcousticConfig = PitchLatticeAcousticConfig(),
    ) -> None:
        super().__init__()
        self.config = config
        trellis_config = config.trellis_config()
        self.duration = TrellisDurationPredictor(trellis_config)
        self.acoustic = TrellisAcousticPath(trellis_config)
        self.envelope = nn.Conv1d(
            config.acoustic_width,
            config.spectral_features,
            kernel_size=1,
        )
        self.detail = FactorizedTemporalConv1d(
            config.acoustic_width,
            config.spectral_features,
            config.detail_kernel_size,
            config.detail_rank,
        )
        self.detail_gate = nn.Conv1d(config.acoustic_width, 1, kernel_size=1)
        self.excitation = FactorizedTemporalConv1d(
            config.acoustic_width,
            4,
            config.control_kernel_size,
            config.control_rank,
        )

    def forward(
        self,
        ids: torch.Tensor,
        durations: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if ids.ndim != 2 or ids.shape[1] <= 0:
            raise ValueError("PitchLattice IDs must have non-empty [batch, tokens] shape")
        mask = torch.ones_like(ids, dtype=torch.bool)
        log_durations = self.duration(ids, mask)
        used_durations = (
            self.duration.durations_from_prediction(log_durations)
            if durations is None
            else durations.to(device=ids.device, dtype=torch.long)
        )
        hidden = self.acoustic(ids, used_durations)
        envelope = self.envelope(hidden)
        detail = self.detail(hidden)
        detail_gate = torch.sigmoid(self.detail_gate(hidden))
        spectral = envelope + detail_gate * detail
        raw = self.excitation(hidden)
        excitation = torch.cat(
            [
                1.5 * torch.tanh(raw[:, 0:1]),
                torch.sigmoid(raw[:, 1:2]),
                torch.sigmoid(raw[:, 2:3]),
                torch.tanh(raw[:, 3:4]),
            ],
            dim=1,
        )
        return {
            "controls": torch.cat([spectral, excitation], dim=1),
            "spectral": spectral,
            "envelope": envelope,
            "detail": detail,
            "detail_gate": detail_gate,
            "excitation": excitation,
            "hidden": hidden,
            "log_durations": log_durations,
            "durations": used_durations,
        }

    @torch.no_grad()
    def initialize_trellises(self, source: TrellisRift) -> None:
        source_config = source.config
        target_config = self.config.trellis_config()
        ignored = {
            "n_mels",
            "head_steps",
            "head_kernel_size",
            "head_expansion",
            "output_kernel_size",
        }
        source_values = {
            key: value for key, value in source_config.to_dict().items() if key not in ignored
        }
        target_values = {
            key: value for key, value in target_config.to_dict().items() if key not in ignored
        }
        if source_values != target_values:
            raise ValueError("source Trellis-RIFT backbone configuration changed")
        self.duration.load_state_dict(source.duration.state_dict(), strict=True)
        self.acoustic.load_state_dict(source.acoustic.state_dict(), strict=True)


def pitch_lattice_acoustic_parameter_breakdown(
    model: PitchLatticeAcoustic,
) -> dict[str, int]:
    values = {
        "duration": sum(parameter.numel() for parameter in model.duration.parameters()),
        "acoustic": sum(parameter.numel() for parameter in model.acoustic.parameters()),
        "envelope": sum(parameter.numel() for parameter in model.envelope.parameters()),
        "detail": sum(parameter.numel() for parameter in model.detail.parameters()),
        "detail_gate": sum(parameter.numel() for parameter in model.detail_gate.parameters()),
        "excitation": sum(parameter.numel() for parameter in model.excitation.parameters()),
    }
    values["front_total"] = int(sum(values.values()))
    return {name: int(value) for name, value in values.items()}


__all__ = [
    "PitchLatticeAcoustic",
    "PitchLatticeAcousticConfig",
    "pitch_lattice_acoustic_parameter_breakdown",
]
