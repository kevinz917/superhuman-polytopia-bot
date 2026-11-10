#!/bin/zsh
# Session E launcher: reserve the container-hour bound in the ledger, then start a detached-from-shell modal call.
#   launch.sh train NAME SECONDS TIMEOUT USD  [extra modal train args...]
#   launch.sh arena NAME MATCHES_JSON_FILE TIMEOUT USD
set -e
cd /Users/Shared/PolytopiaV3/wt-v3-compute
kind=$1; name=$2
M=(env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy /Users/Shared/PolytopiaClaude/.venv/bin/modal run --env staging)
out=/Users/Shared/PolytopiaV3/results/$name
if [[ $kind == train ]]; then
  seconds=$3; timeout=$4; usd=$5; shift 5
  python3 experiments/claude_v3/ledger.py reserve $name $usd "1xH100 4CPU/32GiB ${seconds}s training, timeout ${timeout}s"
  nohup $M experiments/claude_v3/modal_v3.py::train --run-id $name --config experiments/claude_v3/hero-1v1-classic.json \
    --seconds $seconds --timeout $timeout --out $out.json "$@" > $out.log 2>&1 &
elif [[ $kind == arena ]]; then
  file=$3; timeout=$4; usd=$5
  python3 experiments/claude_v3/ledger.py reserve $name $usd "1xH100 arena timeout ${timeout}s"
  nohup $M experiments/claude_v3/modal_v3.py::arena --timeout $timeout --matches-json "$(cat $file)" --out $out.json > $out.log 2>&1 &
fi
echo "launched $kind $name"
