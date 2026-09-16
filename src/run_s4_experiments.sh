#!/bin/bash
# s4 (視覚 + 32ch生波形) の学習+評価を 4タスク x 2方策 x 3シード で回す。
#
# 1シード分 (8本) の学習+評価が全て終わってから次のシードに進む。
# 学習・評価の設定は train0.sh / train1.sh / train2.sh に厳密に合わせている:
#   ACT       : batch_size=8
#   Diffusion : batch_size=32 --policy.use_separate_rgb_encoder_per_camera=true
#   steps     : sound / soundDiff        = 100000 (save_freq 10000)
#               soundShake / soundSim    = 200000 (save_freq 20000)
#   eval      : src/eval_policy.py --training-name <run> --checkpoint-step <steps>
#               (episode-num 100, 224x224 は eval_policy.py の既定値)
#
# 既に checkpoint と success_rate.txt が揃っている組み合わせはスキップするので、
# 中断しても同じコマンドで再開できる。
set -u
cd "$(dirname "$0")/.."

GPUS="${GPUS:-0,1}"
SLOTS_PER_GPU="${SLOTS_PER_GPU:-2}"
SEEDS=(${SEEDS:-0 1 2})
LOG_ROOT="${LOG_ROOT:-outputs/s4_exp/logs}"
mkdir -p "$LOG_ROOT"

# wandb は未ログインだと学習が落ちるため既定は無効。
# 使う場合は `uv run wandb login` の後に WANDB=1 を指定する。
WANDB="${WANDB:-0}"
if [ "$WANDB" = "1" ]; then
  WANDB_ARGS="--wandb.enable=true --wandb.disable_artifact=true"
else
  WANDB_ARGS="--wandb.enable=false"
fi

UV="uv run --no-sync"

# task:steps:save_freq  (長いジョブから先に流すとスロットの詰まりが少ない)
TASKS=(
  "soundShake-m4-f10-s4-p0_0:200000:20000"
  "soundSim-m4-f10-s4-p0_0:200000:20000"
  "sound-m4-f10-s4-p0_0:100000:10000"
  "soundDiff-m4-f10-s4-p0_0:100000:10000"
)
POLICIES=(diffusion act)

policy_args() {  # $1 = policy
  case "$1" in
    act)       echo "--batch_size=8" ;;
    diffusion) echo "--batch_size=32 --policy.use_separate_rgb_encoder_per_camera=true" ;;
    *) echo "unknown policy: $1" >&2; exit 1 ;;
  esac
}

for seed in "${SEEDS[@]}"; do
  jobs_file="${LOG_ROOT}/seed${seed}.jobs"
  : > "$jobs_file"

  for entry in "${TASKS[@]}"; do
    ds="${entry%%:*}"; rest="${entry#*:}"
    steps="${rest%%:*}"; save_freq="${rest##*:}"
    step_dir=$(printf "%06d" "$steps")

    for policy in "${POLICIES[@]}"; do
      name="${policy}_${ds}_seed${seed}"
      ckpt="outputs/train/${name}/checkpoints/${step_dir}/pretrained_model"
      result="outputs/eval/${name}_${step_dir}/success_rate.txt"

      if [ -f "$result" ]; then
        echo "[s4] skip (評価済み): ${name}"
        continue
      fi

      train_cmd="${UV} lerobot-train \
--dataset.repo_id=local/${ds} \
--dataset.root=datasets/${ds} \
--policy.type=${policy} \
--output_dir=outputs/train/${name} \
--job_name=${name} \
--policy.device=cuda \
--policy.push_to_hub=false \
${WANDB_ARGS} \
--dataset.video_backend=pyav \
$(policy_args "$policy") \
--steps=${steps} \
--save_freq=${save_freq} \
--seed=${seed}"
      eval_cmd="${UV} python src/eval_policy.py --training-name ${name} --checkpoint-step ${steps}"

      if [ -d "$ckpt" ]; then
        # 学習済みで評価だけ残っている場合
        cmd="$eval_cmd"
      else
        cmd="${train_cmd} && ${eval_cmd}"
      fi
      printf '%s\t%s\n' "$name" "$cmd" >> "$jobs_file"
    done
  done

  n=$(grep -cve '^[[:space:]]*$' "$jobs_file" || true)
  if [ "$n" -eq 0 ]; then
    echo "[s4] seed${seed}: 実行するジョブはありません"
    continue
  fi

  echo "=============================================================="
  echo "[s4] seed${seed}: ${n} 本を開始 (gpus=${GPUS}, slots/gpu=${SLOTS_PER_GPU})"
  echo "=============================================================="
  bash noise_exp/run_jobs.sh "$jobs_file" "$GPUS" "$SLOTS_PER_GPU" "${LOG_ROOT}/seed${seed}"
  echo "[s4] seed${seed} 完了"
done

echo "[s4] 全シード終了。結果:"
for f in outputs/eval/*s4-p0_0_seed*/success_rate.txt; do
  [ -f "$f" ] || continue
  echo "  $(basename "$(dirname "$f")") : $(head -1 "$f")"
done
