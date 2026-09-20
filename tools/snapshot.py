"""快速验证抓帧通道：找到游戏窗口，两种方式各抓一帧，报告耗时与画面统计。

用法::

    .venv\\Scripts\\python.exe tools\\snapshot.py
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402

from d4tracker import capture  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def stats(img):
    small = img[::16, ::16].reshape(-1, 3).astype("int16")
    return small.mean(), small.std(axis=0).max()


def main() -> int:
    wins = capture.list_windows()
    print(f"枚举到 {len(wins)} 个可见顶层窗口")

    game = capture.find_game_window()
    if game is None:
        print("\n没找到游戏窗口。当前屏幕上的候选：")
        for w in sorted(wins, key=lambda w: w.width * w.height, reverse=True)[:12]:
            print(f"  {os.path.basename(w.exe):32s} {w.title[:40]!r}  {w.width}x{w.height}")
        return 1

    print(f"\n游戏窗口: {game}")

    out = os.path.join(ROOT, "samples")
    os.makedirs(out, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")

    print()
    for method in ("printwindow", "bitblt"):
        try:
            t0 = time.perf_counter()
            img = capture.grab_window(game, method=method)
            dt = (time.perf_counter() - t0) * 1000.0
        except Exception as exc:                      # noqa: BLE001
            print(f"  {method:12s} 失败: {exc}")
            continue

        mean, std = stats(img)
        blank = capture._looks_blank(img)
        path = os.path.join(out, f"py-{method}-{stamp}.png")
        cv2.imwrite(path, img)
        print(f"  {method:12s} {img.shape[1]}x{img.shape[0]}  {dt:7.1f} ms  "
              f"亮度={mean:6.1f}  通道std={std:6.1f}  空白={blank}"
              f"  -> {os.path.basename(path)}")

    print("\n连拍 30 帧（printwindow）测耗时分布：")
    ts = []
    for _ in range(30):
        t0 = time.perf_counter()
        capture.grab_window(game, method="printwindow")
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts.sort()
    med = ts[len(ts) // 2]
    print(f"  min={ts[0]:.1f}ms  中位={med:.1f}ms  p90={ts[int(len(ts) * 0.9)]:.1f}ms  "
          f"max={ts[-1]:.1f}ms   => 中位下约 {1000.0 / med:.1f} fps 上限")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
