#!/bin/zsh
# Hero runs (docs/hero-runs-plan-2026-10-26.md). Usage, from the worktree root:
#   experiments/hero/launch.sh run 1v1|ffa4         # learner + actor fleet; writes runs/hero-1026/t0-<run>
#   experiments/hero/launch.sh followups 1v1|ffa4   # ladders or 4p monitor, fleet keeper, stop backstop (after the gates)
# T0: env T0, else runs/hero-1026/t0-<run> (set it to the learner's real start before followups).
set -u
cd "${0:A:h}/../.."
APP=experiments/modal_distributed_rl.py
LEDGER=runs/hero-1026
SECONDS_TRAIN=${SECONDS_TRAIN:-32400}  # training seconds (LR decay ends here)
LADDER_INTERVAL=${LADDER_INTERVAL:-7200}
LADDER_ROUNDS=${LADDER_ROUNDS:-4}
ACTOR_GPU=${ACTOR_GPU:-A10G}
PROCS=48
SERVE='{"trunk_dtype": "bfloat16", "server_cores": 2}'   # 2 pinned cores for the inference server (+44% per A10G container, 4p bench 2026-10-28)
ENV='{"POLYTOPIA_MATMUL_PRECISION":"DEFAULT","POLYTOPIA_HEAD_PRECISION":"HIGHEST","POLYTOPIA_TRUNK_DTYPE":"bfloat16","XLA_PYTHON_CLIENT_MEM_FRACTION":".95"}'
CHAMP=/retained/checkpoints/distributed-final-v1/scratch-cont-2-final/final.npz
BN4=/results/distributed/bn-main4/final.npz
C3=/retained/checkpoints/distributed-final-v1/c3-maps-ctl-update-4854/final.npz
FFA4_BIG2=/results/four-player/monitor/ffa4-big2/update-7650.npz
FFA4_CHAMP=/results/four-player/parents/champ-ffa4-2x64/final.npz
mr() { modal run --detach --env staging "$APP::$1" "${@:2}"; }

MODE=$2
RUN=hero-$MODE
case $MODE in
# Sizes from the 2026-10-28 smokes (docs/hero-throughput-2026-10-28.md): 1v1 learner ~210-220k/s on H100s (B200s
# were no faster), 13.6k/s per actor container; 4p learner GPU-bound ~92k/s on 4x H200, 14.7k/s per container.
1v1)  GPU=${GPU:-H100:4}; ACTORS=${ACTORS:-17} ;;
ffa4) GPU=${GPU:-H200:4}; ACTORS=${ACTORS:-7} ;;
*) echo "usage: $0 run|followups 1v1|ffa4"; exit 2 ;;
esac

case "$1" in
run)
  mkdir -p $LEDGER
  date +%s > $LEDGER/t0-$MODE
  mr launch --run-id $RUN --parent /results/hero/$RUN-init.npz --gpu $GPU --learner-cpu 32 \
    --actors $ACTORS --actor-cpu 16 --actor-gpu $ACTOR_GPU --processes $PROCS --seconds $SECONDS_TRAIN \
    --ledger $LEDGER --serve-options-json "$SERVE" --learner-env-json "$ENV" \
    --config-json "$(cat experiments/hero/$RUN.json)" ;;
followups)
  T0=${T0:-$(cat $LEDGER/t0-$MODE)}; NOW=$(date +%s); AGE=$((NOW - T0))
  if [[ $MODE == 1v1 ]]; then
    END=$((T0 + SECONDS_TRAIN + 900))    # final round: 15 min after training_seconds
    mr start_elo_ladder --tag hero-1v1 --run-ids hero-1v1 --interval $LADDER_INTERVAL --rounds $LADDER_ROUNDS \
      --first-delay $((LADDER_INTERVAL - AGE)) \
      --maps 32 --map-mix proposed-v1 --seed-base 470000 \
      --references-json "{\"champion\":\"$CHAMP\",\"bn4\":\"$BN4\",\"c3\":\"$C3\"}" &
    sleep 2
    mr start_elo_ladder --tag hero-1v1-final --run-ids hero-1v1 --rounds 1 --first-delay $((END - NOW)) \
      --maps 64 --map-mix proposed-v1 --seed-base 480000 \
      --references-json "{\"champion\":\"$CHAMP\",\"bn4\":\"$BN4\",\"c3\":\"$C3\"}" &
  else
    mr start_four_monitor --tag hero-ffa4-mon --run-id hero-ffa4 --interval $LADDER_INTERVAL --max-rounds $((LADDER_ROUNDS + 1)) \
      --first-delay $((LADDER_INTERVAL - AGE)) \
      --maps 100 --parts 40 --final-maps 200 --seed-base 490000 --ledger $LEDGER \
      --references-json "{\"START\":\"$FFA4_BIG2\",\"CHAMP\":\"$FFA4_CHAMP\"}" &
  fi
  sleep 2
  mr start_fleet_keeper --run-id $RUN --desired $ACTORS --until-unix $((T0 + SECONDS_TRAIN)) --actor-cpu 16 \
    --processes $PROCS --actor-gpu $ACTOR_GPU --serve-options-json "$SERVE" --ledger $LEDGER &
  sleep 2
  mr schedule_stop --run-id $RUN --at-unix $((T0 + SECONDS_TRAIN + 1800)) &   # backstop only
  wait ;;
*) echo "usage: $0 run|followups 1v1|ffa4"; exit 2 ;;
esac
