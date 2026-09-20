"""分段计时：抓帧 / 模板匹配 / 数值读取 各占多少。"""

import statistics
import sys
import time

sys.path.insert(0, r"D:\deepseek\workspace\d4-tracker")

from d4tracker import capture, detect, values  # noqa: E402

det = detect.Detector()
win = capture.find_game_window()
if not win:
    print("找不到游戏窗口")
    raise SystemExit(1)
print("游戏窗口:", win.width, "x", win.height)

vr = values.ValueReader()
t_cap, t_cls, t_val, t_tot = [], [], [], []

for i in range(20):
    t0 = time.perf_counter()
    img = capture.grab_window(win, method="auto", copy=False)   # 与热路径一致
    t1 = time.perf_counter()
    det.classify(img)
    t2 = time.perf_counter()
    if i % 6 == 0:                       # 与 Monitor 的 value_interval 一致
        vr.read(img)
    t3 = time.perf_counter()

    t_cap.append((t1 - t0) * 1000)
    t_cls.append((t2 - t1) * 1000)
    t_val.append((t3 - t2) * 1000)
    t_tot.append((t3 - t0) * 1000)
    time.sleep(0.33)


def show(name, xs, every=1):
    ys = xs[::every]
    print(f"{name:14s} 均值 {statistics.mean(ys):7.1f} ms   中位 {statistics.median(ys):7.1f} ms   "
          f"最大 {max(ys):7.1f} ms")


show("抓帧", t_cap)
show("模板匹配", t_cls)
show("数值读取(每6帧)", t_val, 6)
show("合计", t_tot)
print()
per_frame = statistics.mean(t_tot) / 1000.0
print(f"每帧 {per_frame * 1000:.1f} ms；3fps 下单核占用约 "
      f"{per_frame * 3 * 100:.1f}%")
