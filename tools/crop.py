"""从样本里裁剪指定区域并按帧纵向拼接，用来读原生分辨率的文案。

联络表能把整段过程缩成一张图，但 1440p 压到 480px/格之后文字就没法看了。
判定必须建立在准确文案上，所以还需要"固定一块区域、跨几帧纵排"的视图：

* 区域用像素坐标给（左上原点，相对游戏窗口）
* 每帧一条，左侧标帧号；必要时整体放大，方便认小字
* 输出宽度控制在 ~1000px 以内，避免被再压一次

用法::

    .venv\\Scripts\\python.exe tools\\crop.py --dir 梦魇地下城-完成-20260920-210543 ^
        --box 300,60,1000,220 --frames 0,8,16,17,20,24 --out nm-top.png --scale 1.0
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
OUTDIR = os.path.join(SAMPLES, "_crops")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="按区域裁剪并纵排多帧")
    ap.add_argument("--dir", required=True, help="样本目录名（samples 下）")
    ap.add_argument("--box", required=True, help="x,y,w,h 像素坐标")
    ap.add_argument("--frames", required=True, help="逗号分隔的帧号，例如 0,8,16,17,20")
    ap.add_argument("--out", required=True, help="输出文件名")
    ap.add_argument("--scale", type=float, default=1.0, help="整体缩放，读小字时用 2.0")
    args = ap.parse_args(argv)

    x, y, w, h = (int(v) for v in args.box.split(","))
    frames = [int(v) for v in args.frames.split(",") if v.strip()]

    folder = os.path.join(SAMPLES, args.dir)
    if not os.path.isdir(folder):
        print(f"没有这个样本目录: {folder}")
        return 1

    sw, sh = int(w * args.scale), int(h * args.scale)
    label_w = 90
    canvas = np.zeros((sh * len(frames), sw + label_w, 3), dtype=np.uint8)
    canvas[:] = 24

    for i, idx in enumerate(frames):
        path = os.path.join(folder, f"f{idx:03d}.jpg")
        img = imageio.imread(path)
        if img is None:
            cv2.putText(canvas, f"#{idx}", (8, i * sh + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            continue
        crop = img[y:y + h, x:x + w]
        if crop.size == 0:
            continue
        if args.scale != 1.0:
            crop = cv2.resize(crop, (sw, sh), interpolation=cv2.INTER_CUBIC)
        canvas[i * sh:(i + 1) * sh, label_w:label_w + crop.shape[1]] = crop
        cv2.putText(canvas, f"#{idx}", (8, i * sh + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, f"#{idx}", (8, i * sh + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 255, 80), 1, cv2.LINE_AA)

    os.makedirs(OUTDIR, exist_ok=True)
    out = os.path.join(OUTDIR, args.out)
    if not imageio.imwrite(out, canvas):
        print("写出失败")
        return 1

    print(f"{out}")
    print(f"  源区域 {args.dir}  box=({x},{y},{w},{h})  scale={args.scale}")
    print(f"  帧 {frames}")
    print(f"  输出 {canvas.shape[1]}x{canvas.shape[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
