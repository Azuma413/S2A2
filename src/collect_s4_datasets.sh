#!/bin/bash
# s4 (視覚 + 32ch生波形) データセットを4タスク分収集する。
#
# 既存の s1/s2/s3 データセットと同じ設定・同じエピソード数で新規収集する:
#   sound 100ep / soundDiff 100ep / soundShake 200ep / soundSim 200ep
#
# 出力:
#   datasets/<task>-m4-f10-s4-p0_0/            LeRobotDataset (front/side/state/action
#                                              + sound_source_pos + audio_end_sample)
#   datasets/<task>-m4-f10-s4-p0_0/audio/      エピソード毎の32ch WAV (int16, 16kHz)
#   datasets/<task>-m4-f10-s4-p0_0/meta/audio_info.json
set -eu
cd "$(dirname "$0")/.."

GPUS="${GPUS:-0,1}"
LOG_DIR="${LOG_DIR:-outputs/collect_s4}"
mkdir -p "$LOG_DIR"

# task:episodes
JOBS=(
  "sound-m4-f10-s4-p0:100"
  "soundDiff-m4-f10-s4-p0:100"
  "soundShake-m4-f10-s4-p0:200"
  "soundSim-m4-f10-s4-p0:200"
)

IFS=',' read -r -a GPU_LIST <<< "$GPUS"
pids=()
for i in "${!JOBS[@]}"; do
  task="${JOBS[$i]%%:*}"
  episodes="${JOBS[$i]##*:}"
  gpu="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
  log="${LOG_DIR}/${task}.log"
  echo "[collect] ${task} (${episodes}ep) -> GPU${gpu}, log=${log}"
  CUDA_VISIBLE_DEVICES="$gpu" uv run --no-sync python -u src/make_sim_dataset.py \
    "$task" --episode-num "$episodes" > "$log" 2>&1 &
  pids+=($!)
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done

echo "[collect] 完了 (status=${status})"
for job in "${JOBS[@]}"; do
  task="${job%%:*}"
  for d in datasets/${task}_*; do
    [ -d "$d" ] || continue
    n=$(uv run --no-sync python -c "import json;print(json.load(open('$d/meta/info.json'))['total_episodes'])" 2>/dev/null || echo "?")
    a=$(du -sh "$d/audio" 2>/dev/null | cut -f1)
    echo "  $d : ${n} ep, audio ${a}"
  done
done
exit "$status"
