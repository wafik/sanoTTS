# Indonesian G2P: switch the `id` voice front end to indo-g2p

Date: 2026-08-28
Issue: [Ampixa/sanoTTS#2](https://github.com/Ampixa/sanoTTS/issues/2)
Scope: web demo only (`web/index.html` + assets it loads). Python CLI (`pypkg/`) and `sanotts-web` npm package stay on espeak-ng.

## Problem

espeak-ng's Indonesian support misplaces the schwa on 46.6% of words and emits
glottal stops in 0 of 57 required positions. The `id` voice was trained on
espeak phoneme IDs, so the fix must map into the espeak inventory rather than
retrain.

## Approach (approved)

Phonemize the Indonesian voice in JS with [indo-g2p](https://github.com/snowfluke/indo-g2p)
(MIT, zero deps, `indo-g2p/core` = 235 KB gzipped), map its IPA output onto the
existing `CP_ID_ID` codepoint→phoneme-id table (slot 4), and feed the framed id
sequence into the existing `snt_voice_synthesize` wasm — unchanged. All other
voices keep the espeak path (`snt_g2p_set_voice`). No wasm rebuild, no retrain.

Precedent: `web/trellis_frontend.js` already phonemizes in JS and feeds ids to
wasm for the trellis voice.

## Components

| File | Purpose |
| --- | --- |
| `web/id_g2p.js` | vendored `indo-g2p/core`, esbuild → IIFE global `SaanoIdG2P`. Lazy-loaded only for the `indonesian` voice. Committed like the other built wasm/js artifacts. |
| `web/id_g2p_map.js` | JS copy of `CP_ID_ID` (generated), plus `idG2pToIds(text)`: IPA → stress insertion → codepoint lookup → `[BOS, PAD, (id, PAD)*, EOS]` framing. |
| `tools/gen_id_cp_table.mjs` | extract `CP_ID_ID` from `mcu/ports/wasm/cp_id_tables_multi.h` → JS literal (single source of truth: the C header). |
| `tools/build_id_g2p.sh` | esbuild vendor step that regenerates `web/id_g2p.js`. |
| `tools/id_g2p_eval.mjs` | runnable gate: issue words (rusak, bakso, cerih, bebek, …) must produce `ʔ` and correct `ə` placement; neutral-word parity spot-check vs recorded espeak ids; unknown-codepoint rate must be 0. Non-zero exit on failure. |
| `tools/render_id_samples.py` | re-render `samples-release/indonesian-*.mp3` from saved ids using the numpy models. |
| `web/index.html` | small branch in the synth path: `voice.key === "indonesian"` → lazy-load `id_g2p.js` + `id_g2p_map.js`, call `idG2pToIds`. Fallback to espeak path with a log line if the bundle fails to load. |

## Mapping rules

1. **Stress**: indo-g2p emits no stress marks; the voice was trained on espeak's
   `ˈ`. Insert `ˈ` before the penultimate syllable of each word (default
   Indonesian stress) to approximate the training distribution.
2. **Per-codepoint**: NFD-decompose; every codepoint is looked up individually
   (piper convention). `d`, `ʒ`, `a`, `ɪ` all exist in the table.
3. **Unknown codepoints**: skipped and counted. The eval gate fails if the rate
   is non-zero on the eval set.
4. **Punctuation** (`. , ! ?`) maps directly — they are in the table.
5. **`ɛ`** maps as-is (id 61 exists) — this is part of the schwa improvement.

`CP_ID_ID` already contains `ʔ`(109), `ə`(59), `ɛ`(61), `ŋ`(44), `ˈ`(120) —
the mapping is essentially 1:1.

## Error handling

- Bundle load failure → fall back to the espeak path and log to `#log`; the
  demo never dies.
- `idG2pToIds` on unexpected input (empty string, unknown cp) → skip + count,
  never throw.

## Testing

- `tools/id_g2p_eval.mjs` is the one runnable check (issue-listed words,
  parity spot-check, unknown-cp gate). Run before committing.
- Manual A/B in the browser: "rusak bakso rakyat" before vs after.
- Re-render the two `indonesian-*.mp3` samples with the new ids.

## Out of scope (follow-ups)

- Issue path 2: retrain the `id` voice on indo-g2p phoneme IDs for the full
  schwa gain. The mapping table built here is reused at that point.
- `sanotts-web` npm package and Python CLI keep espeak.
- MCU/ESP32 ports keep espeak (no TS on-device).
