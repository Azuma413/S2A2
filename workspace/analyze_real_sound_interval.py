"""実機データセットの spec 動画から音イベントの発生間隔を推定する。

背景
----
実機データセット（soundReal*）には生波形が保存されていない。音の情報は
`observation.images.spec`（スペクトログラム画像）の動画だけなので、音の強さは
この画像の輝度から読み取るしかない。

spec 画像の横軸は 1.0 秒の解析窓（SoundConfig.processing_time）そのものなので、
1 フレームだけで 1 秒分のエンベロープが読める。そこで

    列方向の平均輝度 -> フレーム内で min-max 正規化 -> ピーク検出

でピーク間隔を求め、これを音イベントの発生間隔とみなす。

しきい値について
----------------
`SoundCamera._convert_spectrogram_to_image` は spectrogram_normalization="percentile"
のとき **フレームごとに** 2〜98 パーセンタイルで正規化する。そのため無音フレームでも
コントラストが最大まで引き伸ばされ、輝度の絶対値は音量を表さない。
固定の輝度しきい値で二値化する方式が成立しないのはこのためで、スケール不変な
prominence（ピークの卓立度）で判定する必要がある。

結果は prominence にほぼ全て支配され、height と distance の影響は小さい。
`--sweep` で感度テーブルを出せるので、値を報告するときは併せて確認すること。

使い方
------
    uv run --no-sync python workspace/analyze_real_sound_interval.py
    uv run --no-sync python workspace/analyze_real_sound_interval.py --sweep
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

# spec 画像の横軸が張る時間 = SoundConfig.processing_time（soundreal_utils.AUDIO_WINDOW_SECONDS）
WINDOW_SECONDS = 1.0
# STFT の時間ビン数: 1 + fs // (nfft // 2) = 1 + 16000 // 256
STFT_BINS = 63

DEFAULT_VIDEO = Path(
    "datasets/soundRealShake-m4-f10-s2-p0/videos/observation.images.spec/chunk-000/file-000.mp4"
)


def probe_video(path: Path) -> dict:
    """ffprobe で解像度とフレーム数を取得する。"""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,nb_frames,r_frame_rate",
            "-of", "json", str(path),
        ],
        stdout=subprocess.PIPE,
        check=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    num, den = stream["r_frame_rate"].split("/")
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "nb_frames": int(stream.get("nb_frames", 0)),
        "fps": float(num) / float(den),
    }


def load_gray_frames(path: Path, step: int, width: int, height: int) -> np.ndarray:
    """step フレームおきにグレースケールで読み込む -> (N, H, W) uint8。

    データセットの動画は AV1 なので OpenCV では開けない。ffmpeg から raw を受け取る。
    """
    result = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(path),
            "-vf", f"select='not(mod(n\\,{step}))'",
            "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "gray", "-",
        ],
        stdout=subprocess.PIPE,
        check=True,
    )
    frame_bytes = width * height
    count = len(result.stdout) // frame_bytes
    if count == 0:
        raise RuntimeError(f"No frames decoded from {path}")
    return np.frombuffer(result.stdout[: count * frame_bytes], dtype=np.uint8).reshape(
        count, height, width
    )


def build_envelopes(frames: np.ndarray) -> np.ndarray:
    """(N, H, W) -> (N, W) の 0..1 エンベロープ。列平均をフレーム内で min-max 正規化する。"""
    profiles = frames.mean(axis=1).astype(np.float32)
    low = profiles.min(axis=1, keepdims=True)
    high = profiles.max(axis=1, keepdims=True)
    span = np.maximum(high - low, 1e-6)
    return np.where(high > low, (profiles - low) / span, 0.0)


def measure_intervals(
    envelopes: np.ndarray, height: float, prominence: float, distance: int
) -> dict:
    """ピーク間隔[秒]を集計する。"""
    seconds_per_px = WINDOW_SECONDS / envelopes.shape[1]
    intervals = []
    peak_count = 0
    for profile in envelopes:
        peaks, _ = find_peaks(
            profile, height=height, prominence=prominence, distance=distance
        )
        peak_count += len(peaks)
        if len(peaks) >= 2:
            intervals.append(np.diff(peaks) * seconds_per_px)

    if not intervals:
        return {
            "height": height,
            "prominence": prominence,
            "distance_px": distance,
            "peaks_per_second": peak_count / len(envelopes),
            "mean_interval_s": None,
            "median_interval_s": None,
            "num_intervals": 0,
        }

    merged = np.concatenate(intervals)
    return {
        "height": height,
        "prominence": prominence,
        "distance_px": distance,
        "peaks_per_second": peak_count / len(envelopes),
        "mean_interval_s": float(merged.mean()),
        "median_interval_s": float(np.median(merged)),
        "num_intervals": int(merged.size),
    }


def print_row(result: dict) -> None:
    mean_s = result["mean_interval_s"]
    median_s = result["median_interval_s"]
    if mean_s is None:
        return
    print(
        f"{result['height']:7.2f} {result['prominence']:11.2f} {result['distance_px']:8d} "
        f"{result['peaks_per_second']:12.2f} {mean_s:12.3f} {median_s:14.3f}"
    )


def print_header() -> None:
    print(
        f"\n{'height':>7} {'prominence':>11} {'dist_px':>8} "
        f"{'peaks/sec':>12} {'mean[s]':>12} {'median[s]':>14}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="実機 spec 動画から音イベントの発生間隔を推定します。"
    )
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO, help="spec 動画のパス。")
    parser.add_argument(
        "--step",
        type=int,
        default=30,
        help="何フレームおきに解析するか。既定の30は動画30fpsに対して1秒ごと。"
        "spec 画像自体が1秒窓なので、これ以上細かくしても情報は増えない。",
    )
    parser.add_argument("--height", type=float, default=0.3, help="ピークの最小高さ（正規化後）。")
    parser.add_argument(
        "--prominence",
        type=float,
        default=0.6,
        help="ピークの最小卓立度。結果を支配するのはこの値。",
    )
    parser.add_argument("--distance", type=int, default=4, help="ピーク間の最小距離[px]。")
    parser.add_argument("--sweep", action="store_true", help="感度テーブルを出力します。")
    parser.add_argument("--json-out", type=Path, default=None, help="結果のJSON保存先。")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    info = probe_video(args.video)
    frames = load_gray_frames(args.video, args.step, info["width"], info["height"])
    envelopes = build_envelopes(frames)

    width = info["width"]
    print(f"video     : {args.video}")
    print(
        f"stream    : {width}x{info['height']}, {info['fps']:.1f}fps, "
        f"{info['nb_frames']} frames"
    )
    print(f"sampled   : {len(frames)} frames (step={args.step})")
    print(
        f"time axis : 1px = {WINDOW_SECONDS / width * 1000:.2f} ms, "
        f"STFT bin = {WINDOW_SECONDS / STFT_BINS * 1000:.2f} ms "
        f"({width / STFT_BINS:.2f} px)"
    )

    if args.sweep:
        results = []
        print_header()
        for height in (0.3, 0.5, 0.6, 0.7):
            for prominence in (0.3, 0.4, 0.5, 0.6):
                for distance in (4, 8, 16):
                    result = measure_intervals(envelopes, height, prominence, distance)
                    results.append(result)
                    print_row(result)
    else:
        result = measure_intervals(envelopes, args.height, args.prominence, args.distance)
        results = [result]
        print_header()
        print_row(result)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps({"video": str(args.video), "info": info, "results": results}, indent=2)
        )
        print(f"\nSaved JSON to {args.json_out}")


if __name__ == "__main__":
    main()
