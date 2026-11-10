#!/bin/zsh
# Session H launcher (session E's launch.sh with H's app, volume and ledger): reserve the container-hour bound in
# /Users/Shared/PolytopiaClaude/v3h-budget.json, then start a detached modal call.
#   launch.sh train NAME SECONDS TIMEOUT USD  [extra modal train args...]
#   launch.sh arena NAME MATCHES_JSON_FILE TIMEOUT USD
set -e
cd /Users/Shared/PolytopiaV3/wt-v3-hybrid-h
kind=$1; name=$2
M=(env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy /Users/Shared/PolytopiaClaude/.venv/bin/modal run --env staging)
out=/Users/Shared/PolytopiaV3/results-h/$name
mkdir -p /Users/Shared/PolytopiaV3/results-h
if [[ $kind == train ]]; then
  seconds=$3; timeout=$4; usd=$5; shift 5
  python3 experiments/claude_v3h/ledger.py reserve $name $usd "1xH100 4CPU/32GiB ${seconds}s training, timeout ${timeout}s"
  nohup $M experiments/claude_v3h/modal_v3h.py::train --run-id $name --config experiments/claude_v3/hero-1v1-classic-nopeace-warm150.json \
    --seconds $seconds --timeout $timeout --out $out.json "$@" > $out.log 2>&1 &
elif [[ $kind == arena ]]; then
  file=$3; timeout=$4; usd=$5
  python3 experiments/claude_v3h/ledger.py reserve $name $usd "1xH100 arena timeout ${timeout}s"
  nohup $M experiments/claude_v3h/modal_v3h.py::arena --timeout $timeout --matches-json "$(cat $file)" --out $out.json > $out.log 2>&1 &
fi
echo "launched $kind $name"
