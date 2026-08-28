#!/bin/sh
# Rebuild web/id_g2p.js (vendored indo-g2p@0.1.2, MIT -- see NOTICE below).
# Requires node/npx on PATH. Pinned: bump the version in BOTH the install and
# this comment together.
#   sh tools/build_id_g2p.sh
set -e
cd "$(dirname "$0")/.."
rm -rf /tmp/idg2p_src && mkdir -p /tmp/idg2p_src
npm pack indo-g2p@0.1.2 --pack-destination /tmp/idg2p_src
tar -xzf /tmp/idg2p_src/indo-g2p-0.1.2.tgz -C /tmp/idg2p_src
npx esbuild tools/id_g2p_entry.js \
  --bundle --minify --format=iife \
  --alias:indo-g2p/core=/tmp/idg2p_src/package/lib/core.js \
  --outfile=web/id_g2p.js
# license attribution: indo-g2p is MIT; keep its license text beside the bundle
cp /tmp/idg2p_src/package/LICENSE web/id_g2p.LICENSE.txt
ls -la web/id_g2p.js
