/* id_g2p_map.js -- Indonesian front end for the `indonesian` web voice.
 *
 * Indo-g2p emits unmarked IPA; the voice was trained on espeak ids that carry
 * primary stress (ˈ U+02C8). We insert ˈ before the penultimate vowel group of
 * every IPA word (default Indonesian stress), then map each codepoint through
 * the generated CP_ID_ID table and frame Piper-style: BOS, PAD, (id, PAD)*, EOS.
 * Unknown codepoints are skipped and reported, never thrown.
 *
 * Depends on SaanoIdG2P (web/id_g2p.js, indo-g2p core) and SaanoIdTable
 * (web/id_cp_table.js, generated from the C header).
 */
(function (root) {
  "use strict";

  // indo-g2p emits ASCII 'g' (U+0067) in its IPA, but CP_ID_ID maps it to
  // id 154, outside the voice's trained vocab (d_vocab = a_vocab = 122 in
  // web/voices/indonesian/meta.json, so valid ids are 0..121). espeak emits
  // ɡ (U+0261, id 66) for the same phoneme, so rewrite before lookup.
  const CP_REWRITES = new Map([[0x67, 0x261]]);

  // Defense in depth: never emit an id the voice cannot represent. Anything
  // >= VOCAB_MAX + 1 is skipped and reported instead of fed to the models.
  const VOCAB_MAX = 121;

  const VOWELS = /[aeiouɑɛɪɔəʊ]/;

  function insertStress(word) {
    // positions where a vowel group starts
    const starts = [];
    for (let i = 0; i < word.length; i++) {
      if (VOWELS.test(word[i]) && (i === 0 || !VOWELS.test(word[i - 1]))) starts.push(i);
    }
    if (!starts.length) return "ˈ" + word;
    const at = starts.length >= 2 ? starts[starts.length - 2] : 0;
    return word.slice(0, at) + "ˈ" + word.slice(at);
  }

  function textToIds(text) {
    const G2P = root.SaanoIdG2P, Table = root.SaanoIdTable;
    const ipa = G2P.toPhoneme(text, { english: false }).phonemes;
    const stressed = ipa.split(/(\s+)/).map(part =>
      /^\s*$/.test(part) ? part : insertStress(part)).join("");
    const ids = [Table.BOS, Table.PAD];
    const skipped = [];
    for (const ch of stressed.normalize("NFD")) {
      const cp = CP_REWRITES.get(ch.codePointAt(0)) ?? ch.codePointAt(0);
      const id = Table.MAP.get(cp);
      if (id === undefined || id > VOCAB_MAX) { skipped.push({ cp }); continue; }
      ids.push(id, Table.PAD);
    }
    ids.push(Table.EOS);
    return { ids, skipped };
  }

  root.SaanoIdMap = { textToIds };
})(typeof self !== "undefined" ? self : globalThis);
