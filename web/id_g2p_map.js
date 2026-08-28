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
      /\s/.test(part) ? part : insertStress(part)).join("");
    const ids = [Table.BOS, Table.PAD];
    const skipped = [];
    for (const ch of stressed.normalize("NFD")) {
      const cp = ch.codePointAt(0);
      const id = Table.MAP.get(cp);
      if (id === undefined) { skipped.push({ cp }); continue; }
      ids.push(id, Table.PAD);
    }
    ids.push(Table.EOS);
    return { ids, skipped };
  }

  root.SaanoIdMap = { textToIds };
})(typeof self !== "undefined" ? self : globalThis);
