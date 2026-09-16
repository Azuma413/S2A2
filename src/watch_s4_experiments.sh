#!/bin/bash
# s4実験の進捗を監視し、ジョブ完了・失敗・シード完了・ドライバ異常終了を1行ずつ出力する。
# Monitor から呼ばれ、出力の各行がそのまま通知になる。
set -u
cd "$(dirname "$0")/.."

DRIVER_LOG="${DRIVER_LOG:-outputs/s4_exp/driver.log}"
INTERVAL="${INTERVAL:-60}"
PATTERN='DONE|FAIL|\[s4\] seed[0-9] 完了|全シード終了'

# 起動時点で既に出ている行は通知しない
seen=$(grep -cE "$PATTERN" "$DRIVER_LOG" 2>/dev/null || echo 0)

while true; do
  total=$(grep -cE "$PATTERN" "$DRIVER_LOG" 2>/dev/null || echo 0)
  if [ "$total" -gt "$seen" ]; then
    grep -E "$PATTERN" "$DRIVER_LOG" | tail -n +$((seen + 1)) | while IFS= read -r line; do
      # 完了ジョブなら成功率も併せて出す
      name=$(echo "$line" | grep -oE '(act|diffusion)_sound[A-Za-z]*-m4-f10-s4-p0_0_seed[0-9]' || true)
      if [ -n "$name" ]; then
        case "$name" in
          *soundShake*|*soundSim*) step=200000 ;;
          *) step=100000 ;;
        esac
        rate=$(grep -oE '[0-9]+/[0-9]+ \([0-9.]+%\)' \
               "outputs/eval/${name}_$(printf '%06d' $step)/success_rate.txt" 2>/dev/null | head -1)
        [ -n "$rate" ] && line="$line  ->  $rate"
      fi
      echo "$line"
    done
    seen=$total
  fi

  if ! pgrep -f "run_s4_experiments.sh" > /dev/null 2>&1; then
    if grep -q "全シード終了" "$DRIVER_LOG" 2>/dev/null; then
      echo "ALL DONE: 全シードの学習+評価が終了しました"
    else
      echo "ALERT: run_s4_experiments.sh が停止しています（未完了のまま）"
    fi
    exit 0
  fi
  sleep "$INTERVAL"
done
