"""诊断：看看数据库里记了什么，以及当前各模板的实时分数。

用它是为了回答"为什么这次没算上 / 为什么算错了"：

* 数据库里有没有那条记录、分数多少、来源是自动还是手动；
* 游戏在前台时，各模板此刻的匹配分数是多少、离阈值差多远。

用法::

    .venv\\Scripts\\python.exe tools\\status.py
    .venv\\Scripts\\python.exe tools\\status.py --limit 50
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from d4tracker import capture, detect, kinds, paths  # noqa: E402
from d4tracker.store import Store  # noqa: E402

ROOT = paths.app_root()


def show_db(db_path: str, limit: int) -> None:
    if not os.path.exists(db_path):
        print(f"数据库不存在: {db_path}")
        return

    print(f"数据库: {db_path}")
    print(f"  大小: {os.path.getsize(db_path) / 1024:.1f} KB")
    store = Store(db_path)

    rows = store.conn.execute(
        "SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    print(f"\n最近 {len(rows)} 条事件:")
    if not rows:
        print("  （空）")
    for r in rows:
        t = time.strftime("%m-%d %H:%M:%S", time.localtime(r["ts"]))
        src = "自动" if r["source"] == "auto" else "手动"
        score = f"{r['score']:.3f}" if r["source"] == "auto" else "  -  "
        print(f"  {t}  {r['label']:<14s} 分数={score}  连续={r['frames']}帧  {src}")

    print("\n计数（今日）:", {k: v for k, v in store.counts_by_kind().items() if v})
    print("计数（全部）:", {k: v for k, v in store.counts_by_kind("all").items() if v})
    custom = store.custom_counts("all")
    if custom:
        print("自定义条目  :", custom)

    readings = store.conn.execute(
        "SELECT * FROM readings ORDER BY id DESC LIMIT 8").fetchall()
    if readings:
        print("\n数值读数（最新 8 条）:")
        for r in readings:
            t = time.strftime("%m-%d %H:%M:%S", time.localtime(r["ts"]))
            print(f"  {t}  {r['key']:<10s} = {r['value']}")
    for key in ("cinders", "baneful"):
        s = store.readings_summary(key)
        if s["latest"] is not None:
            print(f"  {key}: 最新={s['latest']} 今日峰值={s['max']} "
                  f"今日累计获得={s['gained']} 记录={s['samples']} 次变化")

    print("\n各活动当日/总计:", end=" ")
    today, total = store.counts_by_kind(), store.counts_by_kind("all")
    for kind, label, _ready, _note in kinds.TARGETS:
        print(f"\n  {label:<14s} 今日 {today.get(kind, 0):>3d}   总计 {total.get(kind, 0):>3d}")
    store.close()


def show_live() -> None:
    print("\n" + "=" * 64)
    print("实时分数（需要游戏窗口存在；抓一帧看看各模板离阈值多远）")
    print("=" * 64)

    win = capture.find_game_window()
    if not win:
        print("  没找到游戏窗口——游戏没开，或者用了独占全屏")
        return

    print(f"  游戏窗口: {win.width}x{win.height}  前台={capture.is_foreground(win.hwnd)}")
    det = detect.Detector()
    try:
        img = capture.grab_window(win, method="auto")
    except OSError as exc:
        print(f"  抓帧失败: {exc}")
        return

    print(f"  阈值 = {detect.MATCH_THRESHOLD}（连续 {detect.CONFIRM_FRAMES} 帧才算一次）\n")
    for name, label, iou in det.classify(img):
        tpl = next((t for t in det.templates if t.name == name), None)
        where = f"strip={tpl.strip}" if (tpl and tpl.search) else f"roi={tpl.roi if tpl else '?'}"
        flag = "  <<< 超过阈值" if iou >= detect.MATCH_THRESHOLD else ""
        print(f"  {label:<14s} {iou:.3f}   {where}{flag}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="D4 副本计数器 诊断")
    ap.add_argument("--db", default=paths.db_path())
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--no-live", action="store_true", help="不抓帧，只看数据库")
    args = ap.parse_args(argv)

    show_db(args.db, args.limit)
    if not args.no_live:
        show_live()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
