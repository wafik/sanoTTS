#!/usr/bin/env bash
# Build the Trellis-RIFT text-to-wave model to WebAssembly.
#
# Produces web/snt_trellis.js (+ .wasm) exporting SaanoTrellis: a MODULARIZE'd
# Emscripten module wrapping mcu/src/snt_trellis.c (phoneme ids -> durations ->
# acoustic trellis -> recurrent head -> low-rank TinyVocos -> iSTFT -> DC
# blocker -> 24 kHz float PCM) behind the handle API in snt_trellis_wasm.c.
#
# No model weights are baked in or preloaded: the page fetches
# artifacts/trellis-web-port-20260819/bundle/trellis_rift_995058_bundle.bin
# (3,982,096 bytes = meta.bin ++ 995,058 float32) at runtime and passes the raw
# bytes to snt_trellis_wasm_run.
#
# Memory: FIXED at 256MB, no ALLOW_MEMORY_GROWTH, for the same reason as
# build_voices.sh -- JS holds typed-array views into HEAPF32/HEAP32 across
# malloc-heavy calls, and a growth-triggered realloc detaches them. 256MB
# covers roughly 4000 frames (~43 s of speech); beyond that the largest
# buffers (the [1026, T] STFT head output and the [513, T] spectrum) dominate
# and snt_trellis_wasm_run reports -4 (allocation failure) rather than
# corrupting anything. Long inputs need the chunked design sketched in
# section 6 of docs/trellis-web-port-operators.md, which must be re-gated
# against fresh fixtures because chunking changes the noise seeding.
#
# Requires Emscripten (emcc) on PATH: https://emscripten.org
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
mcu="$(cd "$here/../.." && pwd)"          # .../mcu
repo="$(cd "$mcu/.." && pwd)"             # repo root
web="$repo/web"

command -v emcc >/dev/null 2>&1 || { echo "emcc not found on PATH -- install Emscripten" >&2; exit 1; }

src="$mcu/src/snt_trellis.c"
[ -f "$src" ] || { echo "missing $src" >&2; exit 1; }

mkdir -p "$web"

emcc \
  -O3 -std=c99 \
  -I"$mcu/include" \
  "$src" \
  "$here/snt_trellis_wasm.c" \
  -sMODULARIZE=1 -sEXPORT_NAME=SaanoTrellis \
  -sINITIAL_MEMORY=268435456 \
  -sEXPORTED_FUNCTIONS='_malloc,_free,_snt_trellis_wasm_run,_snt_trellis_wasm_release,_snt_trellis_wasm_tokens,_snt_trellis_wasm_frames,_snt_trellis_wasm_samples,_snt_trellis_wasm_handle_sample_rate,_snt_trellis_wasm_stage,_snt_trellis_wasm_pcm16,_snt_trellis_wasm_sample_rate,_snt_trellis_wasm_seed_key' \
  -sEXPORTED_RUNTIME_METHODS='cwrap,HEAPU8,HEAP16,HEAP32,HEAPF32' \
  -sENVIRONMENT=web,worker,node \
  -o "$web/snt_trellis.js"

echo "built $web/snt_trellis.js ($(wc -c <"$web/snt_trellis.wasm") bytes wasm)"
echo "verify with:  node $here/verify_trellis_node.mjs"
