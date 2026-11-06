#!/bin/sh
set -eu
cd "$(dirname "$0")/../../../.."
if test -f "$HOME/.config/polytopia-skirmish/openrouter.key"; then
  OPENROUTER_API_KEY=$(cat "$HOME/.config/polytopia-skirmish/openrouter.key")
  export OPENROUTER_API_KEY
fi
name="continuous-split-$(date -u +%Y%m%dT%H%M%SZ)"
colima start --profile polytopia-skirmish
exec artifacts/harbor-runtime/bin/python scripts/run_skirmish_split.py \
  --name "$name" --directory "artifacts/harbor-campaigns/$name" \
  --local-package artifacts/harbor-packages/split-local-release-v6-20261031 \
  --daytona-package artifacts/harbor-packages/split-daytona-release-v6-20261031 \
  --attempts 5 --local-concurrency 32 --daytona-concurrency 64 --probe --run
