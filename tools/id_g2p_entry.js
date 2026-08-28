// esbuild entry: expose only indo-g2p/core as a browser global. Never import
// the package root here -- it carries the 2 MB English table and 3.4 MB POS
// model that the core entry point exists to avoid.
import { toPhoneme, toSyllables, VERSION } from "indo-g2p/core";
globalThis.SaanoIdG2P = { toPhoneme, toSyllables, VERSION };
