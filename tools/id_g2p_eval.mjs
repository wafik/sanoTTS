// id_g2p_eval.mjs -- runnable gate for the Indonesian indo-g2p front end.
// Fails on: wrong glottal/schwa output on issue-listed words, wrong framing,
// unmapped codepoints in the eval sentences. Exit 0 = pass.
//   node tools/id_g2p_eval.mjs
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import vm from "node:vm";

const here = dirname(fileURLToPath(import.meta.url));
const web = resolve(here, "../web");

// load the two browser scripts the way index.html does (classic globals)
const ctx = { console, globalThis: {} };
vm.createContext(ctx);
vm.runInContext(readFileSync(resolve(web, "id_g2p.js"), "utf8"), ctx);
vm.runInContext(readFileSync(resolve(web, "id_cp_table.js"), "utf8"), ctx);
vm.runInContext(readFileSync(resolve(web, "id_g2p_map.js"), "utf8"), ctx);
const G2P = ctx.globalThis.SaanoIdG2P, Table = ctx.globalThis.SaanoIdTable, M = ctx.globalThis.SaanoIdMap;
if (!G2P) throw new Error("SaanoIdG2P missing -- run tools/build_id_g2p.sh");
if (!Table) throw new Error("SaanoIdTable missing -- run tools/gen_id_cp_table.mjs");
if (!M) throw new Error("SaanoIdMap missing");

let fails = 0;
const check = (name, cond, detail) => {
  if (cond) console.log("PASS", name);
  else { fails++; console.error("FAIL", name, detail ?? ""); }
};

// --- issue #2 measurements: glottal stops + schwa placement ---------------
const GLOTTAL = ["rusak", "bakso", "rakyat", "bebek", "rek"];
for (const w of GLOTTAL) {
  const p = G2P.toPhoneme(w, { english: false }).phonemes;
  check(`glottal in ${w}`, p.includes("ʔ"), p);
}
const SCHWA = { cerih: "tʃərih", dengan: "dəŋan", pergi: "pərgi" };
for (const [w, want] of Object.entries(SCHWA)) {
  const p = G2P.toPhoneme(w, { english: false }).phonemes;
  check(`schwa in ${w}`, p === want, `${p} != ${want}`);
}

// --- framing: BOS PAD (id PAD)* EOS, ids resolve, stress present ----------
const CASES = [
  "Selamat pagi! Hari ini cuaca sangat cerah.",
  "Halo! Senang bertemu dengan Anda hari ini.",
  "Pagi ini cuaca cerah, mari kita mulai hari dengan semangat.",
  "rusak bakso rakyat.",
];
for (const s of CASES) {
  const { ids, skipped } = M.textToIds(s);
  check(`framing ${s.slice(0, 20)}…`,
    ids[0] === 1 && ids[1] === 0 && ids[ids.length - 1] === 2 && ids.length > 4,
    JSON.stringify(ids.slice(0, 8)));
  check(`no gaps ${s.slice(0, 20)}…`, ids.every((v, i) => i % 2 === 1 ? v === 0 : true));
  check(`no skipped cps ${s.slice(0, 20)}…`, skipped.length === 0,
    skipped.map(x => "U+" + x.cp.toString(16)).join(","));
  const stressCount = ids.filter(v => v === 120).length; // ˈ = U+02C8
  const wordCount = s.split(/\s+/).length;
  check(`stress per word ${s.slice(0, 20)}…`, stressCount === wordCount,
    `${stressCount} vs ${wordCount} words`);
}

// --- monosyllable stress lands at word start ------------------------------
const one = M.textToIds("ke").ids; // "ke" -> 1 vowel
check("monosyllable stress", one.includes(120));

// --- edge whitespace: no stress on empty segments --------------------------
const edge = M.textToIds(" Selamat pagi ").ids; // 2 real words + edge blanks
const edgeStress = edge.filter(v => v === 120).length;
check("edge whitespace stress", edgeStress === 2, `${edgeStress} stress marks`);
check("edge whitespace framing", edge[0] === 1 && edge[1] === 0 && edge[edge.length - 1] === 2);
// --- ASCII 'g' rewrite + out-of-vocab guard --------------------------------
// indo-g2p emits ASCII 'g' (U+0067) -> CP_ID_ID id 154, outside the voice's
// vocab (d_vocab = a_vocab = 122; valid ids 0..121). espeak emits ɡ (U+0261)
// -> id 66, which IS in vocab; the map must rewrite and never emit >= 122.
const gs = M.textToIds("gajah gempa pagi");
check("g rewrite to ɡ id 66", gs.ids.includes(66), JSON.stringify(gs.ids));
check("no ASCII g id 154", !gs.ids.includes(154), JSON.stringify(gs.ids));
check("no out-of-vocab ids", gs.ids.every(v => v < 122), "ids >= 122 present");
const dq = M.textToIds('"');
check("double-quote skipped", dq.skipped.length === 1 && dq.skipped[0].cp === 0x22,
  JSON.stringify(dq));
check("double-quote not emitted", !dq.ids.includes(150), JSON.stringify(dq.ids));

const empty = M.textToIds("").ids;
check("empty input framing", empty.length === 3 && empty[0] === 1 && empty[1] === 0 && empty[2] === 2,
  JSON.stringify(empty));
check("empty input no stress", !empty.includes(120), JSON.stringify(empty));

if (fails) { console.error(`\n${fails} check(s) FAILED`); process.exit(1); }
console.log("\nall checks passed");

// --dump-ids artifacts/id_ids.json: emit new-pipeline ids for the renderer
if (process.argv.includes("--dump-ids")) {
  mkdirSync("artifacts", { recursive: true });
  const out = {};
  for (const s of CASES) out[s] = { ids: M.textToIds(s).ids };
  writeFileSync("artifacts/id_ids.json", JSON.stringify(out, null, 1));
  console.log("dumped artifacts/id_ids.json");
}
