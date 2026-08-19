// verify_trellis_frontend_node.mjs -- parity gate for the Trellis-RIFT
// browser frontend.
//
// The Trellis-RIFT English model consumes a character-level tokenization of a
// Misaki-style IPA phoneme string over a frozen 62-symbol vocabulary. The
// browser has to reproduce, exactly, what the Python packaging path produces:
//
//   phonemizer EspeakBackend(en-us, preserve_punctuation, with_stress, tie)
//     -> misaki.espeak.EspeakFallback  (the E2M rewrite table)
//       -> kokoro_frontend.phonemes_to_token_ids
//
// This loads web/snt_g2p.js (export snt_g2p_text_to_ipa) plus
// web/trellis_frontend.js, runs them over trellis_frontend_corpus.json, and
// compares against the Python reference produced by
// tools/trellis_frontend_reference.py. Both the phoneme STRING and the final
// token IDs must match exactly, and a sentence the Python side rejects (e.g.
// over the 207-token limit) must be rejected by the browser side too.
//
// This is achievable only because the espeak-ng bundled at
// mcu/ports/wasm/espeak is PACKAGE_VERSION 1.52.0, the same version the
// Python reference uses. If either side's espeak-ng version moves, expect
// this gate to fail and re-derive the reference before touching the port.
//
// Generate the reference (on a box with torch + misaki, NOT the 18GB Mac):
//   scp mcu/ports/wasm/trellis_frontend_corpus.json <box>:~/…/corpus.json
//   ssh <box> 'cd ~/saanotts-if-vocos && PYTHONPATH=rift-code/src:rift-code/tools:staging-trellis:tools \
//       frontend-venv/bin/python pyref.py'   # tools/trellis_frontend_reference.py
//   scp <box>:~/saanotts-if-vocos/pyref.json /tmp/pyref.json
//
// Then run, from the web/ directory (snt_g2p.js resolves snt_g2p.data
// relative to the process CWD):
//   cd web && node ../mcu/ports/wasm/verify_trellis_frontend_node.mjs /tmp/pyref.json
//
// Exit 0 when every sentence matches, 1 otherwise.
import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import vm from "node:vm";

const require = createRequire(import.meta.url);
const here = dirname(fileURLToPath(import.meta.url));
const repo = resolve(here, "../../..");
const web = resolve(repo, "web");

const refPath = process.argv[2];
if (!refPath) {
  console.error("usage: node verify_trellis_frontend_node.mjs <pyref.json>");
  process.exit(2);
}

vm.runInThisContext(readFileSync(resolve(web, "trellis_frontend.js"), "utf8"));
const Frontend = globalThis.SaanoTrellisFrontend;
if (!Frontend) {
  console.error("FAIL: trellis_frontend.js did not define SaanoTrellisFrontend");
  process.exit(1);
}

const factory = require(resolve(web, "snt_g2p.js"));
const Module = await factory();
const frontend = Frontend.createFrontend({ module: Module });

const reference = JSON.parse(readFileSync(refPath, "utf8"));
let phonemeMatches = 0;
let idMatches = 0;
let matchedRejections = 0;
const failures = [];

for (const row of reference) {
  const { text } = row;
  const pyRejected = "phoneme_error" in row || "id_error" in row;

  let phonemes = null;
  let jsRejected = false;
  let jsError = null;
  try {
    phonemes = frontend.phonemize(text);
  } catch (err) {
    jsRejected = true;
    jsError = `${err.kind || "error"}: ${err.message}`;
  }

  let ids = null;
  if (!jsRejected) {
    try {
      ids = frontend.phonemesToTokenIds(phonemes).ids;
    } catch (err) {
      jsRejected = true;
      jsError = `${err.kind || "error"}: ${err.message}`;
    }
  }

  if (pyRejected && jsRejected) {
    matchedRejections += 1;
    phonemeMatches += 1;
    idMatches += 1;
    continue;
  }
  if (pyRejected !== jsRejected) {
    failures.push({ text, why: "rejection mismatch", python: row.phoneme_error || row.id_error || "accepted", js: jsError || "accepted" });
    continue;
  }
  if (phonemes === row.phonemes) {
    phonemeMatches += 1;
  } else {
    failures.push({ text, why: "phoneme string", python: row.phonemes, js: phonemes });
  }
  const same = Array.isArray(ids) && Array.isArray(row.ids)
    && ids.length === row.ids.length && ids.every((v, i) => v === row.ids[i]);
  if (same) {
    idMatches += 1;
  } else {
    failures.push({ text, why: "token ids", python: row.ids, js: ids });
  }
}

const n = reference.length;
console.log(`sentences:                  ${n}`);
console.log(`phoneme-string exact match: ${phonemeMatches}/${n}`);
console.log(`token-ID exact match:       ${idMatches}/${n} (incl. ${matchedRejections} matched rejections)`);

if (failures.length) {
  console.error(`\nFAIL: ${failures.length} divergence(s)`);
  for (const f of failures.slice(0, 20)) {
    console.error(`  [${f.why}] ${JSON.stringify(f.text)}`);
    console.error(`      python: ${JSON.stringify(f.python)}`);
    console.error(`      js:     ${JSON.stringify(f.js)}`);
  }
  process.exit(1);
}
console.log("\nPASS: browser frontend reproduces the Python contract exactly.");
