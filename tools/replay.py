"""把样本当录像回放，验证状态机每段只出一次事件。

样本的价值就在这里：每段"完成"的地面真值是明确的 —— **恰好 1 次**。
横幅停留 4~14 帧，如果状态机写错（比如按帧累加），回放会立刻把错误放大成
"一次完成记了 14 笔"。

时间轴取自样本的 meta.json（记录着每帧相对时间），所以释放间隔、冷却这些
按秒计的参数能在回放里如实生效。

用法::

    .venv\\Scripts\\python.exe tools\\replay.py
    .venv\\Scripts\\python.exe tools\\replay.py --verbose
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from d4tracker import detect, imageio, kinds  # noqa: E402
from d4tracker.counter import EventCounter  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES = os.path.join(ROOT, "samples")

NEGATIVE_PREFIXES = ("反例", "手动抓帧", "测试样本")


def frame_times(folder: str, count: int) -> list[float]:
    """从 meta.json 取每帧相对时间；没有就按 3fps 估。"""
    meta = os.path.join(folder, "meta.json")
    if os.path.exists(meta):
        try:
            with open(meta, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            ts = [f["t_rel"] for f in data.get("frames", [])]
            if len(ts) == count:
                return ts
        except (OSError, KeyError, ValueError):
            pass
    return [i / 3.0 for i in range(count)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="回放样本验证状态机")
    ap.add_argument("--verbose", action="store_true", help="打印每帧分数")
    args = ap.parse_args(argv)

    det = detect.BannerDetector()
    if not det.ready:
        print("没有模板，先跑 tools/train.py")
        return 1

    names = det.names()

    folders = sorted(
        d for d in os.listdir(SAMPLES)
        if os.path.isdir(os.path.join(SAMPLES, d)) and not d.startswith("_") and d != "crops"
    )

    print(f"阈值={detect.MATCH_THRESHOLD}  确认帧数={detect.CONFIRM_FRAMES} "
          f"释放间隔=1.0s  冷却=20s")
    print("=" * 74)

    rows = []
    for name in folders:
        folder = os.path.join(SAMPLES, name)
        files = sorted(f for f in os.listdir(folder) if f.startswith("f") and f.endswith(".jpg"))
        if not files:
            continue

        times = frame_times(folder, len(files))
        counter = EventCounter(threshold=detect.MATCH_THRESHOLD,
                               confirm_frames=detect.CONFIRM_FRAMES)

        for i, fn in enumerate(files):
            img = imageio.imread(os.path.join(folder, fn))
            if img is None:
                continue
            scores = {t: s for t, _l, s in det.classify(img)}
            fired = counter.update(scores, now=times[i])
            if args.verbose:
                top = max(scores.items(), key=lambda kv: kv[1])
                mark = "  <== 触发" if fired else ""
                print(f"    #{i:03d} t={times[i]:5.2f} {top[0]}={top[1]:.3f}{mark}")

        expected = kinds.expected_kind(name)
        fired_kinds = [ev.kind for ev in counter.events]

        if expected is None:
            verdict = "OK" if not fired_kinds else "不应命中"
        elif fired_kinds == [expected]:
            verdict = "OK"
        elif not fired_kinds:
            verdict = "缺事件"
        elif expected not in fired_kinds:
            verdict = "类型错"
        else:
            verdict = "多事件"

        rows.append((name, expected, fired_kinds, verdict))

        tag = f"期望 {kinds.label_of(expected)}" if expected else "无关"
        print(f"### [{tag}] {name}")
        if counter.events:
            for ev in counter.events:
                mark = "" if expected == ev.kind else "   <- 不是期望的类型"
                print(f"    事件: {kinds.label_of(ev.kind):14s} 分数={ev.score:.3f} "
                      f"连续={ev.frames}帧{mark}")
        else:
            print("    事件: 无")
        print(f"    {verdict}")
        if args.verbose:
            print(counter.report())

    print("\n" + "=" * 74)
    ok = sum(1 for *_, v in rows if v == "OK")
    print(f"完全符合预期: {ok}/{len(rows)}")
    for name, exp, got, verdict in rows:
        if verdict != "OK":
            want = kinds.label_of(exp) if exp else "无"
            have = "/".join(kinds.label_of(k) for k in got) or "无"
            print(f"    {verdict:8s} {name}  (期望 {want} / 实际 {have})")
    return 0 if ok == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
