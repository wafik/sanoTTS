"""SplitBand Trellis acoustic front for LACE-TTS.

The front predicts a smooth mel envelope and a separately gated temporal-detail
residual instead of forcing both through one recurrent mel head. It also emits
pitch, voicing, and aperiodicity controls for the PhaseFlow vocoder.
"""

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
class SplitBandTrellisConfig:
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
    mel_channels: int = 100
    detail_kernel_size: int = 7
    detail_rank: int = 48
    prosody_kernel_size: int = 7
    prosody_rank: int = 16
    max_tokens: int = 207
    min_duration: int = 1
    max_duration: int = 80

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
            n_mels=self.mel_channels,
            max_tokens=self.max_tokens,
            min_duration=self.min_duration,
            max_duration=self.max_duration,
        )


class SplitBandTrellis(nn.Module):
    """Text-to-mel/prosody front with explicit envelope/detail separation."""

    def __init__(self, config: SplitBandTrellisConfig = SplitBandTrellisConfig()) -> None:
        super().__init__()
        self.config = config
        trellis_config = config.trellis_config()
        self.duration = TrellisDurationPredictor(trellis_config)
        self.acoustic = TrellisAcousticPath(trellis_config)
        self.envelope_head = nn.Conv1d(config.acoustic_width, config.mel_channels, 1)
        self.detail_head = FactorizedTemporalConv1d(
            config.acoustic_width,
            config.mel_channels,
            config.detail_kernel_size,
            config.detail_rank,
        )
        self.detail_gate = nn.Conv1d(config.acoustic_width, config.mel_channels, 1)
        self.prosody_head = FactorizedTemporalConv1d(
            config.acoustic_width,
            3,
            config.prosody_kernel_size,
            config.prosody_rank,
        )

    def forward(
        self,
        ids: torch.Tensor,
        durations: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if ids.ndim != 2 or ids.shape[1] <= 0:
            raise ValueError("SplitBand Trellis IDs must have non-empty [B,N] shape")
        mask = torch.ones_like(ids, dtype=torch.bool)
        log_durations = self.duration(ids, mask)
        used_durations = (
            self.duration.durations_from_prediction(log_durations)
            if durations is None
            else durations.to(device=ids.device, dtype=torch.long)
        )
        hidden = self.acoustic(ids, used_durations)
        envelope = self.envelope_head(hidden)
        detail = self.detail_head(hidden)
        detail_gate = torch.sigmoid(self.detail_gate(hidden))
        mel = envelope + detail_gate * detail
        raw_prosody = self.prosody_head(hidden)
        prosody = torch.cat(
            [
                1.5 * torch.tanh(raw_prosody[:, 0:1]),
                torch.sigmoid(raw_prosody[:, 1:2]),
                torch.sigmoid(raw_prosody[:, 2:3]),
            ],
            dim=1,
        )
        return {
            "mel": mel,
            "envelope": envelope,
            "detail": detail,
            "detail_gate": detail_gate,
            "prosody": prosody,
            "hidden": hidden,
            "log_durations": log_durations,
            "durations": used_durations,
        }

    @torch.no_grad()
    def initialize_trellises(self, source: TrellisRift) -> None:
        if source.config != self.config.trellis_config():
            raise ValueError("source Trellis-RIFT configuration changed")
        self.duration.load_state_dict(source.duration.state_dict(), strict=True)
        self.acoustic.load_state_dict(source.acoustic.state_dict(), strict=True)


def splitband_parameter_breakdown(model: SplitBandTrellis) -> dict[str, int]:
    values = {
        "duration": sum(parameter.numel() for parameter in model.duration.parameters()),
        "acoustic": sum(parameter.numel() for parameter in model.acoustic.parameters()),
        "envelope_head": sum(parameter.numel() for parameter in model.envelope_head.parameters()),
        "detail_head": sum(parameter.numel() for parameter in model.detail_head.parameters()),
        "detail_gate": sum(parameter.numel() for parameter in model.detail_gate.parameters()),
        "prosody_head": sum(parameter.numel() for parameter in model.prosody_head.parameters()),
    }
    values["front_total"] = int(sum(values.values()))
    return {name: int(value) for name, value in values.items()}


__all__ = [
    "SplitBandTrellis",
    "SplitBandTrellisConfig",
    "splitband_parameter_breakdown",
]
