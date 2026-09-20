"""样本速览：每组抽帧拼联络表 + 找出帧间变化最大的时刻。

一个个 sample 目录音 30 帧、看 30 张图是不现实的。这里做两件事：

1. **联络表**：每组均匀抽 9 帧，缩到 480x270 拼成 3x3（1920x1080），一眼看完整段
   过程。文件名和帧序号都印在格子上。
2. **变化点**：逐帧算相邻帧的降采样平均绝对差。事件（结算面板弹出）在屏幕上就是
   一次大跳变，所以差值曲线的前几个峰值基本就是事件帧 —— 顺着它去看原图。

用法::

    .venv\\Scripts\\python.exe tools\\contact.py
    .venv\\Scripts\\python.exe tools\\contact.py --only 梦魇
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from d4tracker import imageio  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES = os.path.join(ROOT, "samples")
SHEETS = os.path.join(SAMPLES, "_sheets")

TILE_W, TILE_H = 480, 270
GRID = 3


def frame_paths(folder: str) -> list[str]:
    files = [f for f in os.listdir(folder) if f.startswith("f") and f.endswith(".jpg")]
    return [os.path.join(folder, f) for f in sorted(files)]


def load_small(path: str) -> np.ndarray | None:
    img = imageio.imread(path)
    if img is None:
        return None
    return cv2.resize(img, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA)


def analyse(folder: str) -> list[float]:
    """相邻帧平均绝对差。"""
    smalls = []
    for p in frame_paths(folder):
        s = load_small(p)
        if s is not None:
            smalls.append(s.astype(np.int16))
    diffs = [0.0]
    for a, b in zip(smalls, smalls[1:]):
        diffs.append(float(np.abs(a - b).mean()))
    return diffs


def make_sheet(folder: str, name: str, picks: list[int]) -> str:
    canvas = np.zeros((TILE_H * GRID, TILE_W * GRID, 3), dtype=np.uint8)
    paths = frame_paths(folder)
    for slot, idx in enumerate(picks):
        r, c = divmod(slot, GRID)
        if idx >= len(paths):
            continue
        tile = load_small(paths[idx])
        if tile is None:
            continue
        cv2.putText(tile, f"#{idx}", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(tile, f"#{idx}", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 255, 60), 1, cv2.LINE_AA)
        canvas[r * TILE_H:(r + 1) * TILE_H, c * TILE_W:(c + 1) * TILE_W] = tile

    os.makedirs(SHEETS, exist_ok=True)
    out = os.path.join(SHEETS, f"{name}.jpg")
    imageio.imwrite(out, canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="样本联络表 + 变化点分析")
    ap.add_argument("--only", default=None, help="只处理名字里含该子串的组")
    args = ap.parse_args(argv)

    folders = sorted(
        d for d in os.listdir(SAMPLES)
        if os.path.isdir(os.path.join(SAMPLES, d)) and d != "_sheets" and d != "crops"
    )
    if args.only:
        folders = [d for d in folders if args.only in d]
    if not folders:
        print("没有可处理的样本组")
        return 1

    for name in folders:
        folder = os.path.join(SAMPLES, name)
        diffs = analyse(folder)
        n = len(diffs)
        if n == 0:
            print(f"{name}: 没有帧")
            continue

        # 抽 9 帧：均匀覆盖，另外强制带上差值最大的 3 个位置
        uniform = sorted({round(i * (n - 1) / (GRID * GRID - 1)) for i in range(GRID * GRID)})
        peaks = sorted(range(1, n), key=lambda i: diffs[i], reverse=True)[:3]
        picks = sorted(set(uniform) | set(peaks))

        sheet = make_sheet(folder, name, picks[:GRID * GRID])
        top = ", ".join(f"#{i}({diffs[i]:.1f})" for i in peaks)
        print(f"{name}")
        print(f"    帧数={n}  联络表={os.path.basename(sheet)}")
        print(f"    变化最大: {top}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
