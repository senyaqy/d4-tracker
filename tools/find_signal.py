"""全屏 OCR 找瞬现文案：不预设区域，让信号自己浮出来。

之前只 OCR 顶部一块就判断"库拉斯特/秘语之树/巢穴首领没有完成信号"，那是拿一个
假设当结论。这个工具换个做法：

1. 对一段样本每隔几帧做**全屏** OCR，记录每段文字的内容 + 位置；
2. 按"出现频率"分类 —— 全程都在的是常驻 HUD，**只在少数几帧出现的才是候选信号**；
3. 顺手滤掉纯数字（伤害/等级/金币刷屏）和单字碎片。

用法::

    .venv\\Scripts\\python.exe tools\\find_signal.py --dir 库拉斯特地下城-完成-20260920-205019
    .venv\\Scripts\\python.exe tools\\find_signal.py --all
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from d4tracker import imageio  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES = os.path.join(ROOT, "samples")

_ENGINE = None

# 纯数字/千分位/百分号/加号 —— 伤害数字、金币、经验值刷屏，不是 UI 文案
_NOISE = re.compile(r"^[\d\s,.:%+\-()（）/\\|]*$")


def engine():
    global _ENGINE
    if _ENGINE is None:
        from rapidocr_onnxruntime import RapidOCR
        _ENGINE = RapidOCR()
    return _ENGINE


def ocr_full(img, min_score: float = 0.6):
    """全屏 OCR，返回 [(文本, 分数, (x, y, w, h))]。"""
    res = engine()(img)
    boxes = res[0] if isinstance(res, tuple) else res
    if not boxes:
        return []

    out = []
    for item in boxes:
        try:
            points, text, score = item[0], item[1], item[2]
        except (IndexError, TypeError):
            continue
        text = str(text).strip()
        if not text or float(score) < min_score:
            continue
        if _NOISE.match(text) or len(text) < 2:
            continue
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x, y = int(min(xs)), int(min(ys))
        w, h = int(max(xs) - x), int(max(ys) - y)
        out.append((text, float(score), (x, y, w, h)))
    return out


def analyse(folder: str, step: int, min_score: float):
    files = sorted(f for f in os.listdir(folder) if f.startswith("f") and f.endswith(".jpg"))
    picks = list(range(0, len(files), step))

    seen: dict[str, list[int]] = defaultdict(list)
    where: dict[str, tuple[int, int, int, int]] = {}
    best: dict[str, float] = {}

    for i in picks:
        img = imageio.imread(os.path.join(folder, files[i]))
        if img is None:
            continue
        for text, score, box in ocr_full(img, min_score):
            seen[text].append(i)
            if text not in where or score > best[text]:
                where[text] = box
                best[text] = score

    total = len(picks)
    rows = []
    for text, frames in seen.items():
        rows.append({
            "text": text,
            "n": len(set(frames)),
            "frames": sorted(set(frames)),
            "box": where[text],
            "score": best[text],
            "ratio": len(set(frames)) / max(total, 1),
        })
    rows.sort(key=lambda r: (r["ratio"], -r["n"]))
    return rows, total, picks


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="全屏 OCR 找瞬现文案")
    ap.add_argument("--dir", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--step", type=int, default=3, help="每隔几帧 OCR 一次")
    ap.add_argument("--min-score", type=float, default=0.65)
    ap.add_argument("--persistent", type=float, default=0.8,
                    help="出现频率高于此值算常驻 HUD")
    args = ap.parse_args(argv)

    if args.all:
        dirs = sorted(d for d in os.listdir(SAMPLES)
                      if os.path.isdir(os.path.join(SAMPLES, d))
                      and not d.startswith("_") and d != "crops")
    elif args.dir:
        dirs = [args.dir]
    else:
        ap.error("需要 --dir 或 --all")
        return 1

    for name in dirs:
        folder = os.path.join(SAMPLES, name)
        if not os.path.isdir(folder):
            print(f"没有这个样本目录: {name}")
            continue

        rows, total, picks = analyse(folder, args.step, args.min_score)
        transient = [r for r in rows if r["ratio"] < args.persistent]
        persistent = [r for r in rows if r["ratio"] >= args.persistent]

        print(f"\n### {name}")
        print(f"    采样 {total} 帧: {picks}")
        print(f"    瞬现文案（候选信号） {len(transient)} 条:")
        for r in transient[:16]:
            x, y, w, h = r["box"]
            fr = ",".join(f"#{i}" for i in r["frames"][:10])
            print(f"      {r['text']!r:28s} 出现 {r['n']}/{total} 帧 [{fr}]"
                  f"  位置({x},{y},{w},{h})  分 {r['score']:.2f}")
        if persistent:
            print(f"    常驻 HUD {len(persistent)} 条:")
            for r in persistent[:14]:
                x, y, w, h = r["box"]
                print(f"      {r['text']!r:28s} 出现 {r['n']}/{total} 帧"
                      f"  位置({x},{y},{w},{h})  分 {r['score']:.2f}")
        if not transient and not persistent:
            print("    （这一段里没 OCR 到任何文字）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
