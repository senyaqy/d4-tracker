"""在全部样本上验证检测器：分离度够不够，阈值该定在哪。

方法论（第一版这里是错的）
--------------------------
第一版只把目录名以「反例 / 手动抓帧 / 测试样本」开头的当负样本，结果漏掉了最要命的
一类错误：**正例但类型认错**。实测「炼狱魔潮」的模板在「深坑」样本上打到 0.69~0.74，
因为 `已完成地下城` 其实是通用的地下城完成标记，深坑也有 —— 这种跨类型误报在旧口径
下完全看不见。

正确口径：对每个模板，**负样本 = 所有"期望类型不是它"的目录**（包含其它类型的正例）。

用法::

    .venv\\Scripts\\python.exe tools\\validate.py
    .venv\\Scripts\\python.exe tools\\validate.py --show 0.3
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from d4tracker import detect, imageio, kinds  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES = os.path.join(ROOT, "samples")


def expected_kind(folder: str) -> str | None:
    """期望命中的模板。映射统一放在 d4tracker.kinds，避免两处各写一份而不同步
    （第一版这里自己抄了一份，加了巢穴首领之后忘了同步，直接把正例当成负例）。"""
    return kinds.expected_kind(folder)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="在样本上验证检测器")
    ap.add_argument("--show", type=float, default=0.4, help="打印达到该分数的帧号")
    args = ap.parse_args(argv)

    det = detect.Detector()
    if not det.ready:
        print("没有模板，先跑 tools/train.py")
        return 1

    names = det.names()
    labels = {t.name: t.label for t in det.templates}

    folders = sorted(
        d for d in os.listdir(SAMPLES)
        if os.path.isdir(os.path.join(SAMPLES, d)) and not d.startswith("_") and d != "crops"
    )

    pos_best: dict[str, float] = {n: 0.0 for n in names}
    neg_best: dict[str, float] = {n: 0.0 for n in names}
    neg_src: dict[str, str] = {}

    for name in folders:
        folder = os.path.join(SAMPLES, name)
        files = sorted(f for f in os.listdir(folder) if f.startswith("f") and f.endswith(".jpg"))
        if not files:
            continue

        exp = expected_kind(name)
        best: dict[str, tuple[float, int]] = {n: (0.0, -1) for n in names}
        hits: dict[str, list[int]] = {}

        for i, fn in enumerate(files):
            img = imageio.imread(os.path.join(folder, fn))
            if img is None:
                continue
            for tname, _label, iou in det.classify(img):
                if iou > best[tname][0]:
                    best[tname] = (iou, i)
                if iou >= args.show:
                    hits.setdefault(tname, []).append(i)

        print(f"### [期望 {labels[exp] if exp else '不命中'}] {name}")
        for tname in names:
            score, at = best[tname]
            where = f"#{at}" if at >= 0 else "—"
            shown = ",".join(f"#{i}" for i in hits.get(tname, [])[:12]) or "无"
            flag = ""
            if exp == tname:
                flag = "  <- 正例"
            elif score >= detect.MATCH_THRESHOLD:
                flag = "  !! 误报（超过阈值）"
            print(f"    {labels[tname]:14s} max={score:.3f} @{where:>4s}   >{args.show}: {shown}{flag}")

            if exp == tname:
                pos_best[tname] = max(pos_best[tname], score)
            elif score > neg_best[tname]:
                neg_best[tname] = score
                neg_src[tname] = f"{name}@{where}"

    print("\n" + "=" * 78)
    print(f"{'模板':14s} {'正例最高':>9s} {'负例最高':>9s}   阈值建议")
    bad = []
    for tname in names:
        p, n = pos_best[tname], neg_best[tname]
        if p <= n:
            print(f"{labels[tname]:14s} {p:9.3f} {n:9.3f}   !! 无分离（负例 >= 正例）")
            bad.append(tname)
        else:
            suggest = (p + n) / 2
            ok = "OK" if n < detect.MATCH_THRESHOLD < p else "阈值需调整"
            if ok != "OK":
                bad.append(tname)
            print(f"{labels[tname]:14s} {p:9.3f} {n:9.3f}   {suggest:.2f}  "
                  f"（缝 {p - n:.3f}，当前阈值 {detect.MATCH_THRESHOLD} -> {ok}）")
        if n > 0:
            print(f"{'':14s} {'':9s} {'':9s}   负例最高来自: {neg_src.get(tname, '?')}")

    print()
    print("结论: " + ("全部可分离" if not bad else f"有问题的模板: {[labels[t] for t in bad]}"))
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
