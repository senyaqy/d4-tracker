"""样本采集器：边玩边录，按全局热键把"事件前后几秒"的画面全部落盘。

为什么用环形缓冲
----------------
结算面板、雕文升级界面这类东西只在屏幕上停留几秒。要求玩家"精确在那一瞬间
按热键"是不现实的；保留热键前 N 秒的滚动缓冲，玩家在事件附近随便按一下就能
拿到完整的前后文（这也正是后面做判定状态机时需要的素材）。

存储取舍
--------
1440p 整帧 PNG 约 11MB，缓冲 30 帧就是 330MB —— 不能这么写。环形缓冲里存
JPEG(q=95)，命中时才落盘；命中那一帧额外存一张无损 PNG 供后面抠模板。

用法::

    .venv\\Scripts\\python.exe -m d4tracker.sampler              # 图形窗口
    .venv\\Scripts\\python.exe -m d4tracker.sampler --headless   # 控制台，便于排查
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import capture, imageio
from .hotkeys import HotkeyListener

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT, "config", "hotkeys.json")

DEFAULT_CONFIG = {
    "_说明": "键 = 热键，值 = 标签。改完重启采集器生效。标签会成为样本目录名。",
    "hotkeys": {
        "Ctrl+Alt+1": "梦魇地下城-完成",
        "Ctrl+Alt+2": "深坑-完成",
        "Ctrl+Alt+3": "炼狱魔潮-完成",
        "Ctrl+Alt+4": "库拉斯特地下城-完成",
        "Ctrl+Alt+5": "黑暗堡垒-完成",
        "Ctrl+Alt+6": "秘语之树-交付",
        "Ctrl+Alt+7": "巢穴首领-击杀",
        "Ctrl+Alt+8": "地狱狂潮-余烬面板",
        "Ctrl+Alt+9": "反例-无事件",
        "Ctrl+Alt+0": "随手抓一帧",
    },
    "fps": 3.0,
    "pre_seconds": 6.0,
    "post_seconds": 4.0,
    "jpeg_quality": 95,
    "only_when_foreground": True,
}


def load_config() -> dict:
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    if not os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "w", encoding="utf-8") as fh:
            json.dump(DEFAULT_CONFIG, fh, ensure_ascii=False, indent=2)
        return json.loads(json.dumps(DEFAULT_CONFIG))

    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    for key, val in DEFAULT_CONFIG.items():
        cfg.setdefault(key, val)
    return cfg


def safe_name(text: str) -> str:
    bad = '<>:"/\\|?*'
    out = "".join("_" if c in bad else c for c in text).strip()
    return out or "unnamed"


@dataclass
class Frame:
    t: float
    jpeg: bytes
    method: str


@dataclass
class Stats:
    running: bool = False
    game_found: bool = False
    window_text: str = ""
    frames_captured: int = 0
    frames_skipped: int = 0
    last_fps: float = 0.0
    last_capture_ms: float = 0.0
    buffered: int = 0
    saved_events: int = 0
    last_saved: str = ""
    errors: list[str] = field(default_factory=list)


class Sampler:
    def __init__(self, config: dict, out_root: str) -> None:
        self.cfg = config
        self.out_root = out_root
        self.stats = Stats()
        self._stop = threading.Event()
        self._ring: collections.deque[Frame] = collections.deque(
            maxlen=max(1, int(config["fps"] * config["pre_seconds"])))
        self._lock = threading.Lock()
        self._capture_thread: threading.Thread | None = None
        self._dump_queue: queue.Queue[tuple[str, list[Frame]]] = queue.Queue()
        self._dump_thread: threading.Thread | None = None
        self.hotkeys: HotkeyListener | None = None
        self._win: capture.WindowInfo | None = None
        self._last_hot_frame: tuple[str, np.ndarray] | None = None

    # ------------------------------------------------------------ 采集

    def start(self) -> None:
        self._stop.clear()
        self.stats.running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, name="capture", daemon=True)
        self._capture_thread.start()
        self._dump_thread = threading.Thread(target=self._dump_loop, name="dump", daemon=True)
        self._dump_thread.start()

        self.hotkeys = HotkeyListener(self.cfg["hotkeys"])
        self.hotkeys.start()

    def stop(self) -> None:
        self._stop.set()
        if self.hotkeys:
            self.hotkeys.stop()
        if self._capture_thread:
            self._capture_thread.join(timeout=5.0)
        self._dump_queue.put(("", []))          # 唤醒 dump 线程退出
        if self._dump_thread:
            self._dump_thread.join(timeout=10.0)
        self.stats.running = False

    def _capture_loop(self) -> None:
        interval = 1.0 / float(self.cfg["fps"])
        quality = int(self.cfg["jpeg_quality"])
        only_fg = bool(self.cfg["only_when_foreground"])
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]

        # 前 5 帧的耗时用来修正节拍，避免采集本身把节奏拖垮
        while not self._stop.is_set():
            t0 = time.perf_counter()

            win = self._win
            if win is None or not self._window_alive(win):
                win = capture.find_game_window()
                self._win = win
                self.stats.game_found = win is not None
                self.stats.window_text = str(win) if win else ""

            if win is None:
                self.stats.frames_skipped += 1
                self._sleep_remaining(t0, interval)
                continue

            if only_fg and not capture.is_foreground(win.hwnd):
                self.stats.frames_skipped += 1
                self._sleep_remaining(t0, interval)
                continue

            try:
                img = capture.grab_window(win, method="auto")
            except OSError as exc:
                self.stats.errors.append(f"抓帧失败: {exc}")
                self.stats.errors = self.stats.errors[-5:]
                self._sleep_remaining(t0, interval)
                continue

            cap_ms = (time.perf_counter() - t0) * 1000.0
            ok, buf = cv2.imencode(".jpg", img, encode_params)
            if not ok:
                self.stats.errors.append("JPEG 编码失败")
                self.stats.errors = self.stats.errors[-5:]
                self._sleep_remaining(t0, interval)
                continue

            now = time.time()
            with self._lock:
                self._ring.append(Frame(t=now, jpeg=buf.tobytes(),
                                        method=capture.last_method()))
                self.stats.buffered = len(self._ring)

            self.stats.frames_captured += 1
            self.stats.last_capture_ms = cap_ms
            total_ms = (time.perf_counter() - t0) * 1000.0
            self.stats.last_fps = 1000.0 / max(total_ms, 1.0)

            self._poll_hotkeys()
            self._sleep_remaining(t0, interval)

    @staticmethod
    def _window_alive(win: capture.WindowInfo) -> bool:
        return capture.is_window(win.hwnd)

    @staticmethod
    def _sleep_remaining(t0: float, interval: float) -> None:
        rest = interval - (time.perf_counter() - t0)
        if rest > 0:
            time.sleep(rest)

    # ------------------------------------------------------------ 热键

    def _poll_hotkeys(self) -> None:
        if not self.hotkeys:
            return
        while True:
            try:
                label = self.hotkeys.queue.get_nowait()
            except queue.Empty:
                return
            with self._lock:
                pre = list(self._ring)
            self._dump_queue.put((label, pre))

    # ------------------------------------------------------------ 落盘

    def _dump_loop(self) -> None:
        post_count = max(1, int(self.cfg["fps"] * self.cfg["post_seconds"]))
        interval = 1.0 / float(self.cfg["fps"])

        while not self._stop.is_set() or not self._dump_queue.empty():
            try:
                label, pre = self._dump_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if not label and not pre:
                return

            stamp = time.strftime("%Y%m%d-%H%M%S")
            folder = os.path.join(self.out_root, f"{safe_name(label)}-{stamp}")
            os.makedirs(folder, exist_ok=True)

            # 热键之后继续录 post 秒，把"事件之后"的画面也拿到
            post: list[Frame] = []
            for _ in range(post_count):
                if self._stop.is_set():
                    break
                t0 = time.perf_counter()
                win = self._win
                if win is not None:
                    try:
                        img = capture.grab_window(win, method="auto")
                        ok, buf = cv2.imencode(".jpg", img,
                                               [int(cv2.IMWRITE_JPEG_QUALITY), int(self.cfg["jpeg_quality"])])
                        if ok:
                            post.append(Frame(t=time.time(), jpeg=buf.tobytes(),
                                              method=capture.last_method()))
                    except OSError:
                        pass
                self._sleep_remaining(t0, interval)

            frames = pre + post
            times = [f.t for f in frames]
            base = times[0] if times else time.time()

            manifest = {
                "label": label,
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "hotkey_frame_index": len(pre) - 1,
                "frame_count": len(frames),
                "fps": self.cfg["fps"],
                "pre_seconds": self.cfg["pre_seconds"],
                "post_seconds": self.cfg["post_seconds"],
                "window": self.stats.window_text,
                "frames": [],
            }

            written: list[dict] = []
            for i, fr in enumerate(frames):
                arr = cv2.imdecode(np.frombuffer(fr.jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if arr is None:
                    continue

                # 路径含中文，必须走 imageio（cv2.imwrite 在这里会静默失败）
                name = f"f{i:03d}.jpg"
                if not imageio.imwrite(os.path.join(folder, name), arr,
                                       [int(cv2.IMWRITE_JPEG_QUALITY), int(self.cfg["jpeg_quality"])]):
                    continue

                entry = {"file": name, "t_rel": round(fr.t - base, 3), "method": fr.method}

                # 热键命中那一帧额外存一张无损 PNG，供后面抠模板
                if i == len(pre) - 1:
                    png = "hotkey-lossless.png"
                    if imageio.imwrite(os.path.join(folder, png), arr):
                        entry["lossless"] = png

                written.append(entry)

            manifest["frames"] = written
            manifest["frame_count"] = len(written)

            with open(os.path.join(folder, "meta.json"), "w", encoding="utf-8") as fh:
                json.dump(manifest, fh, ensure_ascii=False, indent=2)

            self.stats.saved_events += 1
            self.stats.last_saved = folder

    # ------------------------------------------------------------ 手动

    def manual_snapshot(self, label: str = "手动抓帧") -> None:
        with self._lock:
            pre = list(self._ring)
        self._dump_queue.put((label, pre))


# ---------------------------------------------------------------- 图形界面


def run_gui(sampler: Sampler) -> int:
    from PySide6 import QtCore, QtWidgets

    app = QtWidgets.QApplication(sys.argv)
    win = QtWidgets.QWidget()
    win.setWindowTitle("D4 样本采集器")
    win.resize(560, 460)
    win.setWindowFlags(win.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)

    layout = QtWidgets.QVBoxLayout(win)

    rec = QtWidgets.QLabel("● 采集中")
    rec.setStyleSheet("color:#c00;font-size:20px;font-weight:bold;")
    layout.addWidget(rec)

    info = QtWidgets.QLabel("")
    info.setStyleSheet("font-family:Consolas,monospace;")
    info.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
    layout.addWidget(info)

    hk = QtWidgets.QLabel("")
    hk.setStyleSheet("font-family:Consolas,monospace;")
    hk.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
    layout.addWidget(hk)

    warn = QtWidgets.QLabel("")
    warn.setStyleSheet("color:#b60;")
    warn.setWordWrap(True)
    layout.addWidget(warn)

    row = QtWidgets.QHBoxLayout()
    btn_snap = QtWidgets.QPushButton("手动抓一帧")
    btn_open = QtWidgets.QPushButton("打开样本目录")
    row.addWidget(btn_snap)
    row.addWidget(btn_open)
    layout.addLayout(row)

    layout.addWidget(QtWidgets.QLabel(f"样本目录：{sampler.out_root}"))

    def refresh():
        s = sampler.stats
        info.setText(
            f"游戏窗口 : {'已找到' if s.game_found else '未找到'}\n"
            f"采集帧数 : {s.frames_captured}  (跳过 {s.frames_skipped})\n"
            f"实际帧率 : {s.last_fps:5.1f} fps   单帧 {s.last_capture_ms:5.1f} ms\n"
            f"缓冲深度 : {s.buffered} 帧 ≈ {s.buffered / max(sampler.cfg['fps'], 0.1):.1f} 秒\n"
            f"已存事件 : {s.saved_events}\n"
            f"最后保存 : {os.path.basename(s.last_saved) if s.last_saved else '—'}"
        )
        lines = ["热键："]
        if sampler.hotkeys:
            for spec, label in sampler.hotkeys.registered:
                lines.append(f"  {spec:14s} -> {label}")
            for spec, why in sampler.hotkeys.failed:
                lines.append(f"  {spec:14s} !! {why}")
        hk.setText("\n".join(lines))
        if s.errors:
            warn.setText("最近错误：" + " / ".join(s.errors[-3:]))
        elif sampler.hotkeys and sampler.hotkeys.failed:
            warn.setText("有热键登记失败，多半和别的程序冲突，改 config/hotkeys.json 换个组合。")
        else:
            warn.setText("")

    btn_snap.clicked.connect(lambda: sampler.manual_snapshot())
    btn_open.clicked.connect(lambda: os.startfile(sampler.out_root))

    timer = QtCore.QTimer()
    timer.timeout.connect(refresh)
    timer.start(500)

    sampler.start()
    refresh()
    win.show()
    code = app.exec()
    sampler.stop()
    return code


# ---------------------------------------------------------------- 控制台


def run_headless(sampler: Sampler, test_dump_after: float | None = None) -> int:
    sampler.start()
    print("=" * 66)
    print("D4 样本采集器（控制台模式）")
    print("=" * 66)
    print(f"游戏窗口 : {'已找到' if sampler.stats.game_found else '未找到'}")
    print(f"样本目录 : {sampler.out_root}")
    print(f"帧率     : {sampler.cfg['fps']} fps   缓冲 {sampler.cfg['pre_seconds']}s + 事后 {sampler.cfg['post_seconds']}s")
    print(f"仅前台   : {sampler.cfg['only_when_foreground']}")
    print("\n热键：")
    if sampler.hotkeys:
        for spec, label in sampler.hotkeys.registered:
            print(f"  {spec:14s} -> {label}")
        for spec, why in sampler.hotkeys.failed:
            print(f"  {spec:14s} !! {why}")
    print("\nCtrl+C 结束。\n")

    started = time.time()
    triggered = False
    last_events = -1
    try:
        while True:
            time.sleep(0.5)

            if test_dump_after is not None and not triggered and (time.time() - started) >= test_dump_after:
                triggered = True
                print(f"[{time.strftime('%H:%M:%S')}] 自检：触发一次落盘")
                sampler.manual_snapshot("测试样本")

            s = sampler.stats
            if s.saved_events != last_events:
                last_events = s.saved_events
                if s.last_saved:
                    n = len(os.listdir(s.last_saved)) if os.path.isdir(s.last_saved) else 0
                    print(f"[{time.strftime('%H:%M:%S')}] 已保存 -> {os.path.basename(s.last_saved)}  ({n} 个文件)")
    except KeyboardInterrupt:
        print("\n收尾中 ...")
    finally:
        sampler.stop()
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="D4 样本采集器")
    ap.add_argument("--headless", action="store_true", help="不开图形窗口，走控制台")
    ap.add_argument("--out", default=os.path.join(ROOT, "samples"), help="样本输出目录")
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--pre", type=float, default=None, help="热键前保留秒数")
    ap.add_argument("--post", type=float, default=None, help="热键后继续录秒数")
    ap.add_argument("--ignore-focus", action="store_true",
                    help="不要求游戏在前台（调试用；游戏被遮挡时 BitBlt 只会得到平色）")
    ap.add_argument("--test-dump-after", type=float, default=None,
                    help="N 秒后自动触发一次落盘，自检用，不必真按热键")
    args = ap.parse_args(argv)

    cfg = load_config()
    if args.fps is not None:
        cfg["fps"] = args.fps
    if args.pre is not None:
        cfg["pre_seconds"] = args.pre
    if args.post is not None:
        cfg["post_seconds"] = args.post
    if args.ignore_focus:
        cfg["only_when_foreground"] = False

    os.makedirs(args.out, exist_ok=True)
    sampler = Sampler(cfg, args.out)
    if args.headless:
        return run_headless(sampler, args.test_dump_after)
    return run_gui(sampler)


if __name__ == "__main__":
    raise SystemExit(main())
