"""OCR 探针：把样本里某块区域的文字直接打出来。

看联络表只能认结构、认不了字，而判定规则必须建立在准确文案上。与其一张张
贴原图（既慢又费上下文），不如直接把文字读出来对齐。

用法::

    .venv\\Scripts\\python.exe tools\\ocr_probe.py --dir 深坑-完成-20260920-205936 ^
        --box 1100,30,1460,260 --frames 0,5,10,14,17,20,25
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from d4tracker import imageio  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES = os.path.join(ROOT, "samples")

_ENGINE = None


def engine():
    global _ENGINE
    if _ENGINE is None:
        from rapidocr_onnxruntime import RapidOCR
        _ENGINE = RapidOCR()
    return _ENGINE


def ocr(img) -> list[tuple[str, float]]:
    """返回 [(文本, 置信度)]，按 y 再按 x 排序。"""
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
        ys = [p[1] for p in points]
        xs = [p[0] for p in points]
        out.append((min(ys), min(xs), str(text), float(score)))
    out.sort(key=lambda t: (round(t[0] / 20), t[1]))
    return [(t[2], t[3]) for t in out]


def sweep(args) -> int:
    """把所有样本目录跑一遍，输出"标签 × 帧号 → 文字"的对照表。"""
    dirs = sorted(
        d for d in os.listdir(SAMPLES)
        if os.path.isdir(os.path.join(SAMPLES, d)) and not d.startswith("_") and d != "crops"
    )
    x = y = w = h = 0
    if args.box:
        x, y, w, h = (int(v) for v in args.box.split(","))
    idxs = [int(v) for v in args.frames.split(",") if v.strip()] if args.frames != "all" else None

    engine()          # 先加载，别把加载时间算进第一帧

    for name in dirs:
        folder = os.path.join(SAMPLES, name)
        files = sorted(f for f in os.listdir(folder) if f.startswith("f") and f.endswith(".jpg"))
        picks = idxs if idxs else [0, len(files) // 3, 2 * len(files) // 3, len(files) - 1]

        print(f"\n### {name}")
        for i in picks:
            if i >= len(files):
                continue
            img = imageio.imread(os.path.join(folder, files[i]))
            if img is None:
                continue
            if args.box:
                img = img[y:y + h, x:x + w]
            if img.size == 0:
                continue
            if args.scale != 1.0:
                import cv2
                img = cv2.resize(img, None, fx=args.scale, fy=args.scale,
                                 interpolation=cv2.INTER_CUBIC)
            lines = [t for t, s in ocr(img) if s >= args.min_score]
            print(f"  #{i:03d}  {' | '.join(lines) if lines else '(无文字)'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="对样本区域做 OCR 并打印文字")
    ap.add_argument("--dir", default=None, help="样本目录名；用 --sweep 时可省略")
    ap.add_argument("--box", default=None, help="x,y,w,h；省略则整帧")
    ap.add_argument("--frames", default="all", help="逗号分隔帧号，或 all")
    ap.add_argument("--min-score", type=float, default=0.5)
    ap.add_argument("--scale", type=float, default=1.0, help="送 OCR 前放大，小字可试 2.0")
    ap.add_argument("--sweep", action="store_true",
                    help="扫所有样本目录（引擎只加载一次，比逐个调用快得多）")
    args = ap.parse_args(argv)

    if args.sweep:
        return sweep(args)

    if not args.dir:
        ap.error("--dir 是必需的（或者用 --sweep 扫全部）")

    folder = os.path.join(SAMPLES, args.dir)
    if not os.path.isdir(folder):
        print(f"没有这个样本目录: {folder}")
        return 1

    names = sorted(f for f in os.listdir(folder) if f.startswith("f") and f.endswith(".jpg"))
    if args.frames == "all":
        idxs = list(range(len(names)))
    else:
        idxs = [int(v) for v in args.frames.split(",") if v.strip()]

    if args.box:
        x, y, w, h = (int(v) for v in args.box.split(","))

    print(f"样本 {args.dir}")
    print(f"区域 {args.box or '整帧'}   scale={args.scale}")
    print("=" * 70)

    for i in idxs:
        if i >= len(names):
            continue
        img = imageio.imread(os.path.join(folder, names[i]))
        if img is None:
            print(f"#{i:03d}  读取失败")
            continue
        if args.box:
            img = img[y:y + h, x:x + w]
        if img.size == 0:
            print(f"#{i:03d}  区域为空")
            continue
        if args.scale != 1.0:
            import cv2
            img = cv2.resize(img, None, fx=args.scale, fy=args.scale,
                             interpolation=cv2.INTER_CUBIC)

        t0 = time.perf_counter()
        lines = [(t, s) for t, s in ocr(img) if s >= args.min_score]
        ms = (time.perf_counter() - t0) * 1000.0

        joined = " | ".join(t for t, _ in lines) if lines else "(无文字)"
        print(f"#{i:03d} [{ms:6.1f}ms] {joined}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
