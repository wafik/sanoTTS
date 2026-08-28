"""Render before/after WAVs for the indonesian voice front end (issue #2).

before: pypkg engine as-is (espeak ids)
after:  ids dumped by `node tools/id_g2p_eval.mjs --dump-ids` (indo-g2p ids)

    node tools/id_g2p_eval.mjs --dump-ids
    python tools/render_id_samples.py
Output: artifacts/id_g2p_samples/<n>-{before,after}.wav (22.05 kHz mono).

Run from the repo root so the relative artifact paths land in ./artifacts/.
The Synthesizer downloads/caches the `id` voice into ~/.cache/sanotts on
first use. Requires only numpy (the pypkg has zero required deps beyond it).
"""
import json
from pathlib import Path

import numpy as np

from sanotts.engine import Synthesizer

IDS_JSON = Path("artifacts/id_ids.json")
OUT = Path("artifacts/id_g2p_samples")


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    import wave
    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as h:
        h.setnchannels(1)
        h.setsampwidth(2)
        h.setframerate(sample_rate)
        h.writeframes(pcm16.tobytes())


def main() -> None:
    if not IDS_JSON.is_file():
        raise SystemExit(
            f"{IDS_JSON} not found -- run `node tools/id_g2p_eval.mjs --dump-ids` first"
        )
    cases = json.loads(IDS_JSON.read_text(encoding="utf-8"))
    if not cases:
        raise SystemExit(f"{IDS_JSON} is empty -- nothing to render")
    OUT.mkdir(parents=True, exist_ok=True)
    synth = Synthesizer("id")  # downloads/caches the id voice
    for n, (text, obj) in enumerate(cases.items(), 1):
        after = synth.synthesize_from_ids(obj["ids"])
        write_wav(OUT / f"{n}-after.wav", after.audio, after.sample_rate)
        before = synth.synthesize(text)
        write_wav(OUT / f"{n}-before.wav", before.audio, before.sample_rate)
        print(f"clip {n}: {text[:40]}...")
    print(f"wrote {len(cases) * 2} wavs to {OUT}/")


if __name__ == "__main__":
    main()
