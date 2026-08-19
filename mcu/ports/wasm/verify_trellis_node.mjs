// verify_trellis_node.mjs -- stage-by-stage gate for the Trellis-RIFT C/WASM
// port, run headless under Node.
//
// Loads web/snt_trellis.js (SaanoTrellis, built by build_trellis.sh) plus the
// weight bundle and golden fixtures under
// artifacts/trellis-web-port-20260819/, runs every fixture sentence through
// snt_trellis_wasm_run, and compares each retained intermediate against the
// committed reference tensors. This implements the gating procedure in
// section 7 of docs/trellis-web-port-operators.md.
//
// Which reference each stage is gated against, and why:
//   log_durations, durations, acoustic_hidden,
//   decoder_coordinates, stft_real, stft_imag   -> torch_*.npy
//   decoder_noise, waveform                     -> numpy_*.npy
// The noise is gated against the NumPy reference because torch.randn is not
// bit-reproducible even across its own SIMD levels (section 4 of the spec), so
// the portable contract is plain float32 + libm. The waveform is gated against
// numpy_waveform.npy for the same reason -- it is the output of the reference
// the C port implements.
//
//   bash mcu/ports/wasm/build_trellis.sh
//   node mcu/ports/wasm/verify_trellis_node.mjs                 # all fixtures
//   node mcu/ports/wasm/verify_trellis_node.mjs sentence01      # a subset
//
// Exit 0 when every stage of every fixture clears its threshold, 1 otherwise.
import { readFileSync, readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const repo = resolve(here, "../../..");
const web = resolve(repo, "web");
const artifacts = resolve(repo, "artifacts/trellis-web-port-20260819");
const bundlePath = resolve(artifacts, "bundle/trellis_rift_995058_bundle.bin");
const fixtureRoot = resolve(artifacts, "fixtures");

// Acceptance thresholds, from section 7 of the operator spec.
const PEARSON_MIN = 0.9999;
const WAVEFORM_MAXABS = 1e-3;
const PCM_MAX_LSB = 16;

// stage selectors, must match snt_trellis_wasm.c
const STAGE = {
  LOG_DURATIONS: 0, DURATIONS: 1, ACOUSTIC_HIDDEN: 2, COORDINATES: 3,
  NOISE: 4, SPEC_REAL: 5, SPEC_IMAG: 6, WAVEFORM: 7,
};
const KEEP_ALL = 7;

// ---- minimal .npy reader (v1/v2, little-endian, C order) -------------------

function readNpy(path) {
  const buf = readFileSync(path);
  if (buf.length < 10 || buf.readUInt8(0) !== 0x93 || buf.toString("latin1", 1, 6) !== "NUMPY")
    throw new Error(`${path}: not a .npy file`);
  const major = buf.readUInt8(6);
  let headerLen, headerStart;
  if (major === 1) { headerLen = buf.readUInt16LE(8); headerStart = 10; }
  else if (major === 2) { headerLen = buf.readUInt32LE(8); headerStart = 12; }
  else throw new Error(`${path}: unsupported .npy version ${major}`);
  const header = buf.toString("latin1", headerStart, headerStart + headerLen);
  const descr = /'descr'\s*:\s*'([^']+)'/.exec(header);
  const fortran = /'fortran_order'\s*:\s*(True|False)/.exec(header);
  const shapeText = /'shape'\s*:\s*\(([^)]*)\)/.exec(header);
  if (!descr || !fortran || !shapeText) throw new Error(`${path}: unparsable header ${header}`);
  // torch_stft.npy is written Fortran-ordered (torch's complex tensors keep a
  // column-major memory layout after .numpy()); everything else is C order.
  const isFortran = fortran[1] === "True";
  const shape = shapeText[1].trim() === ""
    ? []
    : shapeText[1].split(",").map((v) => v.trim()).filter((v) => v !== "").map(Number);
  const count = shape.reduce((a, b) => a * b, 1);
  const body = buf.subarray(headerStart + headerLen);
  const dtype = descr[1];
  // Every fixture dtype is little-endian; reject anything else loudly.
  if (dtype[0] === ">") throw new Error(`${path}: big-endian dtype ${dtype}`);
  // Fortran -> C reindex for the 2D case (the only rank the fixtures use it for)
  const toC = (flat) => {
    if (!isFortran || shape.length < 2) return flat;
    if (shape.length !== 2)
      throw new Error(`${path}: fortran order is only supported for rank-2 arrays`);
    const [r, c] = shape;
    const out = new Float64Array(flat.length);
    for (let i = 0; i < r; i++)
      for (let j = 0; j < c; j++) out[i * c + j] = flat[j * r + i];
    return out;
  };
  let data;
  if (dtype === "<f4" || dtype === "=f4") {
    data = Float64Array.from(new Float32Array(body.buffer, body.byteOffset, count));
  } else if (dtype === "<f8" || dtype === "=f8") {
    data = Float64Array.from(new Float64Array(body.buffer, body.byteOffset, count));
  } else if (dtype === "<i8" || dtype === "=i8") {
    const big = new BigInt64Array(body.buffer, body.byteOffset, count);
    data = Float64Array.from(big, (v) => Number(v));
  } else if (dtype === "<i4" || dtype === "=i4") {
    data = Float64Array.from(new Int32Array(body.buffer, body.byteOffset, count));
  } else if (dtype === "<c8" || dtype === "=c8") {
    // interleaved float32 re/im -> two separate real arrays
    const flat = new Float32Array(body.buffer, body.byteOffset, count * 2);
    const re = new Float64Array(count);
    const im = new Float64Array(count);
    for (let i = 0; i < count; i++) { re[i] = flat[2 * i]; im[i] = flat[2 * i + 1]; }
    return { shape, dtype, real: toC(re), imag: toC(im) };
  } else {
    throw new Error(`${path}: unsupported dtype ${dtype}`);
  }
  return { shape, dtype, data: toC(data) };
}

// ---- comparison metrics -----------------------------------------------------

function compare(got, ref) {
  if (got.length !== ref.length)
    throw new Error(`length mismatch: got ${got.length}, reference ${ref.length}`);
  const n = got.length;
  let maxAbs = 0, sa = 0, sb = 0;
  for (let i = 0; i < n; i++) {
    if (!Number.isFinite(got[i])) throw new Error(`non-finite value at index ${i}`);
    const d = Math.abs(got[i] - ref[i]);
    if (d > maxAbs) maxAbs = d;
    sa += got[i];
    sb += ref[i];
  }
  const ma = sa / n, mb = sb / n;
  let num = 0, da = 0, db = 0;
  for (let i = 0; i < n; i++) {
    const x = got[i] - ma, y = ref[i] - mb;
    num += x * y; da += x * x; db += y * y;
  }
  const denom = Math.sqrt(da * db);
  const pearson = denom === 0 ? 1 : num / denom;
  let refMax = 0;
  for (let i = 0; i < n; i++) refMax = Math.max(refMax, Math.abs(ref[i]));
  return { maxAbs, pearson, relative: refMax === 0 ? 0 : maxAbs / refMax, n };
}

function pcm16(values) {
  const out = new Int32Array(values.length);
  for (let i = 0; i < values.length; i++) {
    const v = Math.min(1, Math.max(-1, values[i])) * 32767;
    // round-half-to-even, matching np.rint / C rintf
    const r = Math.round(v);
    out[i] = Math.abs(v - Math.trunc(v)) === 0.5 && r % 2 !== 0 ? r - Math.sign(v) : r;
  }
  return out;
}

// ---- run --------------------------------------------------------------------

const requested = process.argv.slice(2);
const sentences = (requested.length ? requested : readdirSync(fixtureRoot).sort())
  .filter((name) => name.startsWith("sentence"));
if (!sentences.length) {
  console.error(`no fixtures found under ${fixtureRoot}`);
  process.exit(2);
}

const SaanoTrellis = (await import(resolve(web, "snt_trellis.js"))).default;
const M = await SaanoTrellis();

const run = M.cwrap("snt_trellis_wasm_run", "number",
  ["number", "number", "number", "number", "number", "number", "number"]);
const release = M.cwrap("snt_trellis_wasm_release", null, ["number"]);
const stagePtr = M.cwrap("snt_trellis_wasm_stage", "number", ["number", "number"]);
const framesOf = M.cwrap("snt_trellis_wasm_frames", "number", ["number"]);
const tokensOf = M.cwrap("snt_trellis_wasm_tokens", "number", ["number"]);
const samplesOf = M.cwrap("snt_trellis_wasm_samples", "number", ["number"]);
const sampleRateOf = M.cwrap("snt_trellis_wasm_sample_rate", "number", ["number", "number"]);
const seedKeyOf = M.cwrap("snt_trellis_wasm_seed_key", "number", ["number", "number", "number", "number"]);

const bundleBytes = new Uint8Array(readFileSync(bundlePath));
const bundleP = M._malloc(bundleBytes.length);
M.HEAPU8.set(bundleBytes, bundleP);

const declaredRate = sampleRateOf(bundleP, bundleBytes.length);
if (declaredRate <= 0) {
  console.error(`FAIL: snt_trellis_wasm_sample_rate rc=${declaredRate}`);
  process.exit(1);
}
console.log(`bundle: ${bundlePath}`);
console.log(`        ${bundleBytes.length} bytes, sample_rate ${declaredRate} Hz\n`);

function readFloats(ptr, count) {
  if (!ptr) throw new Error("stage pointer is null (stage not retained?)");
  return Float64Array.from(M.HEAPF32.subarray(ptr >> 2, (ptr >> 2) + count));
}
function readInts(ptr, count) {
  if (!ptr) throw new Error("stage pointer is null (stage not retained?)");
  return Float64Array.from(M.HEAP32.subarray(ptr >> 2, (ptr >> 2) + count));
}

let allOk = true;
const summary = [];

for (const name of sentences) {
  const dir = resolve(fixtureRoot, name);
  const input = JSON.parse(readFileSync(resolve(dir, "input.json"), "utf8"));
  const ids = readNpy(resolve(dir, "ids.npy")).data;
  const n = ids.length;

  const idsP = M._malloc(n * 4);
  M.HEAP32.set(Int32Array.from(ids), idsP >> 2);

  // cross-check the seed key the port derives against the fixture's
  const keyCap = n * 12 + 3;
  const keyP = M._malloc(keyCap);
  const keyLen = seedKeyOf(idsP, n, keyP, keyCap);
  const derivedKey = keyLen < 0
    ? null
    : Buffer.from(M.HEAPU8.subarray(keyP, keyP + keyLen)).toString("utf8");
  M._free(keyP);

  const errP = M._malloc(4);
  const t0 = performance.now();
  const handle = run(bundleP, bundleBytes.length, idsP, n, 0, KEEP_ALL, errP);
  const wall = (performance.now() - t0) / 1000;
  const err = M.HEAP32[errP >> 2];
  M._free(errP);
  M._free(idsP);

  if (!handle) {
    console.log(`${name}: FAIL -- snt_trellis_wasm_run returned err=${err}`);
    allOk = false;
    continue;
  }

  const T = framesOf(handle);
  const N = tokensOf(handle);
  const S = samplesOf(handle);
  const rows = [];
  let ok = true;

  const check = (label, got, ref, opts = {}) => {
    let stat;
    try {
      stat = compare(got, ref);
    } catch (e) {
      rows.push({ label, note: `ERROR: ${e.message}`, ok: false });
      ok = false;
      return;
    }
    const exact = opts.exact === true;
    const pass = exact
      ? stat.maxAbs === 0
      : stat.pearson >= PEARSON_MIN && (opts.maxAbs === undefined || stat.maxAbs <= opts.maxAbs);
    rows.push({ label, ...stat, ok: pass, against: opts.against });
    if (!pass) ok = false;
  };

  if (derivedKey !== input.seed_key) {
    rows.push({ label: "seed_key", note: `derived ${JSON.stringify(derivedKey)} != fixture ${JSON.stringify(input.seed_key)}`, ok: false });
    ok = false;
  }
  if (N !== n || T !== input.frames) {
    rows.push({ label: "shape", note: `tokens ${N} (want ${n}), frames ${T} (want ${input.frames})`, ok: false });
    ok = false;
  }

  check("log_durations", readFloats(stagePtr(handle, STAGE.LOG_DURATIONS), N),
    readNpy(resolve(dir, "torch_log_durations.npy")).data, { against: "torch" });
  check("durations", readInts(stagePtr(handle, STAGE.DURATIONS), N),
    readNpy(resolve(dir, "torch_durations.npy")).data, { against: "torch", exact: true });
  check("acoustic_hidden", readFloats(stagePtr(handle, STAGE.ACOUSTIC_HIDDEN), 96 * T),
    readNpy(resolve(dir, "torch_acoustic_hidden.npy")).data, { against: "torch" });
  check("decoder_coordinates", readFloats(stagePtr(handle, STAGE.COORDINATES), 100 * T),
    readNpy(resolve(dir, "torch_decoder_coordinates.npy")).data, { against: "torch" });
  check("decoder_noise", readFloats(stagePtr(handle, STAGE.NOISE), 4 * T),
    readNpy(resolve(dir, "numpy_decoder_noise.npy")).data, { against: "numpy" });

  const stft = readNpy(resolve(dir, "torch_stft.npy"));
  check("stft_real", readFloats(stagePtr(handle, STAGE.SPEC_REAL), 513 * T), stft.real, { against: "torch" });
  check("stft_imag", readFloats(stagePtr(handle, STAGE.SPEC_IMAG), 513 * T), stft.imag, { against: "torch" });

  const wave = readFloats(stagePtr(handle, STAGE.WAVEFORM), S);
  const refWave = readNpy(resolve(dir, "numpy_waveform.npy")).data;
  check("waveform", wave, refWave, { against: "numpy", maxAbs: WAVEFORM_MAXABS });

  // int16 PCM deviation, the last acceptance criterion in section 7
  const a = pcm16(wave), b = pcm16(refWave);
  let maxLsb = 0, exactCount = 0, within1 = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (d > maxLsb) maxLsb = d;
    if (d === 0) exactCount++;
    if (d <= 1) within1++;
  }
  const pcmOk = maxLsb <= PCM_MAX_LSB;
  if (!pcmOk) ok = false;
  rows.push({
    label: "pcm_int16",
    note: `max ${maxLsb} LSB, ${(100 * exactCount / a.length).toFixed(3)}% bit-identical, ` +
      `${(100 * within1 / a.length).toFixed(3)}% within 1 LSB`,
    ok: pcmOk,
    against: "numpy",
  });

  release(handle);

  const secs = S / declaredRate;
  console.log(`${name}: ${ok ? "PASS" : "FAIL"}  N=${N} T=${T} S=${S} (${secs.toFixed(2)}s audio, ` +
    `synth ${wall.toFixed(3)}s = ${(secs / wall).toFixed(1)}x RT)`);
  for (const r of rows) {
    const mark = r.ok ? " " : "!";
    if (r.note !== undefined) {
      console.log(`  ${mark} ${r.label.padEnd(20)} ${r.note}`);
    } else {
      console.log(`  ${mark} ${r.label.padEnd(20)} vs ${String(r.against).padEnd(5)} ` +
        `max|d|=${r.maxAbs.toExponential(3)}  rel=${r.relative.toExponential(3)}  ` +
        `pearson=${r.pearson.toFixed(10)}`);
    }
  }
  console.log("");
  summary.push({ name, ok });
  if (!ok) allOk = false;
}

M._free(bundleP);

console.log(summary.map((s) => `${s.name}:${s.ok ? "PASS" : "FAIL"}`).join("  "));
console.log(allOk ? "ALL PASS" : "FAIL");
process.exit(allOk ? 0 : 1);
