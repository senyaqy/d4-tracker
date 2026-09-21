"""实时监视：抓帧 -> 检测 -> 状态机 -> 落库，外加一个主界面。

线程模型
--------
采集跑在后台线程（抓帧一帧 20~40ms，不能占住 UI），它只往一个带锁的共享状态里写
"最新一帧的分数 + 预览图 + 新产生的事件"；Qt 侧用 QTimer 定时读出来刷界面。
不做跨线程直接操作控件 —— Qt 里那是崩得最莫名其妙的一类 bug。

用法::

    .venv\\Scripts\\python.exe -m d4tracker                # 主界面
    .venv\\Scripts\\python.exe -m d4tracker --headless      # 控制台，排查用
    .venv\\Scripts\\python.exe -m d4tracker --selftest      # 全链路自检
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time

import cv2
import numpy as np

from . import capture, detect, imageio, kinds, paths
from .counter import Event, EventCounter
from .store import Store
from .values import ValueReader

# 分数落在这个区间算「像但不够」：不是识别成功，但值得留一份现场。
# 定 0.30 是因为实测各模板的反例最高分在 0.16~0.37，从这里往上就有诊断价值了。
NEARMISS_WATCH = 0.30

# 迟滞用的释放阈值：低于它才认为"横幅消失了"。
# 不直接用 MATCH_THRESHOLD(0.55) 是因为分数在阈值附近抖动时，会反复
# "释放 -> 再触发"，把一段常驻文案记成很多次（实战里 90 秒记了 3 次巢穴首领）。
# 0.40 落在实测的干净间隔里：反例最高 0.37，正例最低 0.63。
RELEASE_THRESHOLD = 0.40

ROOT = paths.app_root()
DB_PATH = paths.db_path()


class Monitor:
    """采集 + 判定 + 落库。与界面无关，可单独跑。"""

    def __init__(self, store: Store, detector: detect.Detector,
                 fps: float = 3.0, only_when_foreground: bool = True,
                 value_interval: int = 6) -> None:
        self.store = store
        self.detector = detector
        self.counter = EventCounter(threshold=detect.MATCH_THRESHOLD,
                                    confirm_frames=detect.CONFIRM_FRAMES,
                                    release_threshold=RELEASE_THRESHOLD)
        self.fps = fps
        self.only_when_foreground = only_when_foreground

        # 数值读取（地狱狂潮）两槽要跑 6 次识别，实测单次中位 13ms、最坏 350ms，
        # 和抓帧一个量级。余烬/灾祸之心变化很慢，0.5Hz 足够，所以隔 6 帧读一次。
        self.values = ValueReader()
        self.value_interval = max(1, int(value_interval))
        self._frame_no = 0

        self.lock = threading.Lock()
        self.latest_scores: dict[str, float] = {}
        self.latest_preview: np.ndarray | None = None
        self.new_events: list[Event] = []
        self.latest_values: dict[str, int] = {}
        self.value_confidence: dict[str, float] = {}
        self.value_changes: list[tuple[str, int, float]] = []
        self._last_recorded: dict[str, int] = {}

        # 「接近但没到阈值」的留证。没有这个，事后只能靠猜是漏检还是压根没出现 ——
        # 秘语之树出过一次零识别，当时没有任何现场可查，只能重开一局复现。
        self.nearmiss_best: dict[str, float] = {}
        self.nearmiss_pending: list[tuple[str, str, float, str, str]] = []

        self.frames_seen = 0
        self.frames_skipped = 0
        self.last_capture_ms = 0.0
        self.running = False
        self.game_found = False
        self.window_text = ""
        self.errors: list[str] = []

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._win: capture.WindowInfo | None = None

    # ---------------------------------------------------------------- 核心

    def process_frame(self, img: np.ndarray, now: float | None = None) -> list[Event]:
        """一帧走完全链路。回放和实时都走这里，保证两条路行为一致。"""
        now = time.time() if now is None else now
        results = self.detector.classify(img)
        scores = {t: s for t, _l, s in results}

        fired = self.counter.update(scores, now=now)
        self._check_nearmiss(results, img)
        self._check_short_runs(img)

        # 预览显示"当前得分最高的那个模板"的 ROI —— 出问题时一眼能看出是哪块区域
        best = max(scores.items(), key=lambda kv: kv[1])[0] if scores else None
        roi = self.detector.preview(img, best)
        with self.lock:
            self.latest_scores = scores
            self.latest_preview = roi.copy() if (roi is not None and roi.size) else None
            for ev in fired:
                ev.label = kinds.label_of(ev.kind)
                self.store.record(ev.kind, score=ev.score, frames=ev.frames,
                                  ts=ev.ts, source="auto")
                self.new_events.append(ev)
        return fired

    def drain_events(self) -> list[Event]:
        with self.lock:
            out = self.new_events
            self.new_events = []
        return out

    def drain_nearmiss(self) -> list[tuple[str, str, float, str, str]]:
        with self.lock:
            out = self.nearmiss_pending
            self.nearmiss_pending = []
        return out

    def _save_crop(self, img: np.ndarray, name: str, score: float, tag: str) -> str:
        """把检测器**实际看到的那块 ROI** 存下来，返回路径（失败返回空串）。

        存图是关键：事后看分数只能猜，看图能直接判断是位置偏了、字变了，
        还是提示框压根没出现。
        """
        try:
            roi = self.detector.preview(img, name)
            if roi is None or not roi.size:
                return ""
            out_dir = os.path.join(paths.data_dir(), "nearmiss")
            os.makedirs(out_dir, exist_ok=True)
            stamp = time.strftime("%m%d-%H%M%S")
            path = os.path.join(out_dir, f"{stamp}_{name}_{tag}_{score:.3f}.png")
            if not imageio.imwrite(path, roi):
                return ""
            self._trim_nearmiss(out_dir)
            return path
        except Exception:                                # noqa: BLE001
            return ""

    def _check_nearmiss(self, results: list[tuple[str, str, float]],
                        img: np.ndarray) -> None:
        """分数进了「像但不够」的区间就留一份现场。

        只在刷新了该模板历史最高分（且超过 0.03）时才存，避免同一次事件刷满磁盘。
        复用 process_frame 已经算好的结果，不重复 classify。
        """
        for name, label, iou in results:
            if iou < NEARMISS_WATCH or iou >= detect.MATCH_THRESHOLD:
                continue
            if iou <= self.nearmiss_best.get(name, 0.0) + 0.03:
                continue

            self.nearmiss_best[name] = iou
            path = self._save_crop(img, name, iou, "near")
            with self.lock:
                self.nearmiss_pending.append(
                    (name, label, iou, path,
                     f"最高 {iou:.3f}，未到阈值 {detect.MATCH_THRESHOLD}"))

    def _check_short_runs(self, img: np.ndarray) -> None:
        """命中过阈值、但连续帧数不够 -> 也留证。

        这类是"我明明看到提示了却没算上"的头号嫌疑：3fps 下 confirm_frames=2
        意味着提示得连续显示 0.67 秒。分数明明过了阈值说明模板是对的，
        问题在时长——看图就能确认。
        """
        for kind, best, frames, _ts in self.counter.drain_short_runs():
            path = self._save_crop(img, kind, best, "short")
            need = self.counter.confirm_frames
            with self.lock:
                self.nearmiss_pending.append(
                    (kind, kinds.label_of(kind), best, path,
                     f"命中但只持续 {frames} 帧（需连续 {need} 帧），已忽略"))

    @staticmethod
    def _trim_nearmiss(out_dir: str, keep: int = 60) -> None:
        """只留最近 keep 张，否则玩一晚上能堆出几百兆。"""
        try:
            files = sorted(
                (os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith(".png")),
                key=os.path.getmtime)
            for old in files[:-keep]:
                os.remove(old)
        except OSError:
            pass

    def read_values(self, img: np.ndarray) -> dict[str, int]:
        """读数值型指标（余烬/灾祸之心），只在变化时落库。

        复用同一帧而不是为读数再抓一次：抓一次帧 20~40ms，没必要花两遍。
        """
        current = self.values.read(img)
        with self.lock:
            self.latest_values = dict(current)
            self.value_confidence = dict(self.values.confidence)

        for key, value in current.items():
            if self._last_recorded.get(key) == value:
                continue
            self._last_recorded[key] = value
            self.store.record_reading(key, value)
            with self.lock:
                self.value_changes.append((key, value, time.time()))
        return current

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "scores": dict(self.latest_scores),
                "preview": None if self.latest_preview is None else self.latest_preview.copy(),
                "frames_seen": self.frames_seen,
                "frames_skipped": self.frames_skipped,
                "capture_ms": self.last_capture_ms,
                "running": self.running,
                "game_found": self.game_found,
                "window": self.window_text,
                "active": self.counter.active_kinds(),
                "values": dict(self.latest_values),
                "value_confidence": dict(self.value_confidence),
                "nearmiss": dict(self.nearmiss_best),
                "errors": list(self.errors[-3:]),
            }

    # ---------------------------------------------------------------- 采集

    def start(self) -> None:
        self._stop.clear()
        self.running = True
        self._thread = threading.Thread(target=self._loop, name="monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        self.running = False

    def _loop(self) -> None:
        interval = 1.0 / max(self.fps, 0.2)
        while not self._stop.is_set():
            t0 = time.perf_counter()

            win = self._win
            if win is None or not capture.is_window(win.hwnd):
                win = capture.find_game_window()
                self._win = win
                with self.lock:
                    self.game_found = win is not None
                    self.window_text = str(win) if win else ""

            if win is None:
                with self.lock:
                    self.frames_skipped += 1
                self._sleep(t0, interval)
                continue

            if self.only_when_foreground and not capture.is_foreground(win.hwnd):
                with self.lock:
                    self.frames_skipped += 1
                self._sleep(t0, interval)
                continue

            try:
                # copy=False：这一帧在本轮内用掉就丢，省掉两次整帧拷贝
                img = capture.grab_window(win, method="auto", copy=False)
            except OSError as exc:
                with self.lock:
                    self.errors.append(f"抓帧失败: {exc}")
                self._sleep(t0, interval)
                continue

            with self.lock:
                self.last_capture_ms = (time.perf_counter() - t0) * 1000.0
                self.frames_seen += 1

            self.process_frame(img)

            self._frame_no += 1
            if self._frame_no % self.value_interval == 0:
                self.read_values(img)

            self._sleep(t0, interval)

    @staticmethod
    def _sleep(t0: float, interval: float) -> None:
        rest = interval - (time.perf_counter() - t0)
        if rest > 0:
            time.sleep(rest)


# ---------------------------------------------------------------- 样式

QSS = """
QWidget { font-family: "Microsoft YaHei UI","Microsoft YaHei","Segoe UI";
          font-size: 10.5pt; color: #e6e6ee; }
QWidget#root { background: #13131a; }

QLabel#stateOn  { font-size: 15pt; font-weight: bold; color: #4fc97a; }
QLabel#stateOff { font-size: 15pt; font-weight: bold; color: #7a7a88; }
QLabel#sub   { color: #9a9aa8; font-size: 9.5pt; }
QLabel#hint  { color: #8a8a98; font-size: 9pt; }
QLabel#preview { background: #0c0c11; border: 1px solid #33333f; border-radius: 6px;
                 color: #6a6a78; font-size: 9.5pt; }
QLabel#score { font-family: Consolas,"Cascadia Mono",monospace; font-size: 10pt;
               color: #b9b9c8; }
QLabel#vlabel { font-size: 10pt; color: #cfcfda; }
QLabel#vvalue { font-family: Consolas,"Cascadia Mono",monospace; font-size: 11pt;
                color: #f0c96a; }

QGroupBox { border: 1px solid #2c2c38; border-radius: 9px; margin-top: 16px;
            padding: 12px 10px 8px 10px; font-weight: bold; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; color: #d4575f; }

QPushButton { background: #262633; border: 1px solid #3a3a4a; border-radius: 6px;
              padding: 7px 16px; }
QPushButton:hover   { background: #33333f; border-color: #d4575f; }
QPushButton:pressed { background: #1c1c26; }
QPushButton:disabled{ color: #66666f; border-color: #2a2a34; background: #1e1e26; }
QPushButton#minus { color: #e8a2a6; font-size: 12pt; font-weight: bold; padding: 0; }
QPushButton#plus  { color: #a6dcae; font-size: 12pt; font-weight: bold; padding: 0; }

QTableWidget { background: #17171f; alternate-background-color: #1c1c26;
               border: none; gridline-color: #26262f; }
QTableWidget::item:selected { background: #33333f; }
QHeaderView::section { background: #23232d; padding: 7px; border: none;
                       color: #9a9aa8; font-weight: bold; }
QTableCornerButton::section { background: #23232d; border: none; }

QListWidget { background: #17171f; border: none; }
QTextEdit { background: #0f0f15; border: 1px solid #26262f; border-radius: 6px; }

/* 弹框必须单独上色：全局那条 QWidget{color:浅色} 会作用到 QMessageBox，
   而它的背景仍走系统浅色主题 -> 白字白底，什么都看不见 */
QMessageBox, QInputDialog, QDialog { background: #1b1b24; }
QMessageBox QLabel, QInputDialog QLabel { color: #e6e6ee; font-size: 10.5pt; }
QMessageBox QPushButton, QInputDialog QPushButton { min-width: 84px; padding: 6px 14px; }
QInputDialog QLineEdit, QInputDialog QSpinBox {
    background: #101016; border: 1px solid #33333f; border-radius: 5px;
    padding: 6px; color: #e6e6ee; selection-background-color: #d4575f; }

QSplitter::handle { background: #26262f; }
QSplitter::handle:vertical { height: 4px; }
QSplitter::handle:hover { background: #d4575f; }

QScrollBar:vertical { background: #17171f; width: 11px; margin: 0; }
QScrollBar::handle:vertical { background: #34343f; border-radius: 5px; min-height: 26px; }
QScrollBar::handle:vertical:hover { background: #45454f; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; }
QScrollBar:horizontal { background: #17171f; height: 11px; }
QScrollBar::handle:horizontal { background: #34343f; border-radius: 5px; min-width: 26px; }
"""


def apply_dark_title_bar(widget) -> None:
    """把 Windows 原生标题栏也刷成深色。

    QSS 管不到系统标题栏，深色窗口顶一条白栏很割裂。DwmSetWindowAttribute
    的 20 号属性是 Win11 2004+ 的写法，19 是旧版，两个都试一遍，失败就算了。
    """
    try:
        import ctypes

        hwnd = int(widget.winId())          # 取句柄即建原生窗口，不必先 show
        value = ctypes.c_int(1)
        for attr in (20, 19):
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, attr, ctypes.byref(value), ctypes.sizeof(value))
    except Exception:                        # noqa: BLE001
        pass


def apply_theme(app) -> None:
    """深色主题：调色板 + QSS + 中文翻译。

    只设 QSS 不够 —— 全局那条 ``QWidget{color:浅色}`` 会作用到 QMessageBox，
    而它的背景仍走系统浅色主题，结果白字白底什么都看不见。调色板负责 QSS
    覆盖不到的部分（原生弹框、工具提示、选区）。

    再装 qtbase 的中文翻译，否则 QMessageBox / QInputDialog 的标准按钮是
    Yes / No / OK 而不是「是 / 否 / 确定」。
    """
    from PySide6 import QtCore, QtGui

    app.setStyleSheet(QSS)
    pal = app.palette()
    pal.setColor(QtGui.QPalette.Window, QtGui.QColor("#1b1b24"))
    pal.setColor(QtGui.QPalette.WindowText, QtGui.QColor("#e6e6ee"))
    pal.setColor(QtGui.QPalette.Base, QtGui.QColor("#12121a"))
    pal.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor("#17171f"))
    pal.setColor(QtGui.QPalette.Text, QtGui.QColor("#e6e6ee"))
    pal.setColor(QtGui.QPalette.Button, QtGui.QColor("#262633"))
    pal.setColor(QtGui.QPalette.ButtonText, QtGui.QColor("#e6e6ee"))
    pal.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor("#1b1b24"))
    pal.setColor(QtGui.QPalette.ToolTipText, QtGui.QColor("#e6e6ee"))
    pal.setColor(QtGui.QPalette.Highlight, QtGui.QColor("#d4575f"))
    pal.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor("#ffffff"))
    app.setPalette(pal)

    candidates = [
        os.path.join(paths.resource_root(), "translations", "qtbase_zh_CN.qm"),
        os.path.join(QtCore.QLibraryInfo.path(QtCore.QLibraryInfo.TranslationsPath),
                     "qtbase_zh_CN.qm"),
    ]
    for qm in candidates:
        if not os.path.exists(qm):
            continue
        translator = QtCore.QTranslator(app)      # 挂到 app 上，否则被回收后翻译失效
        if translator.load(qm):
            app.installTranslator(translator)
            break


def _num_item(value: int):
    """计数格子：居中、加粗、金色 —— 一眼能看清的数字才是这个工具的重点。"""
    from PySide6 import QtCore, QtGui, QtWidgets

    item = QtWidgets.QTableWidgetItem(str(value))
    item.setTextAlignment(QtCore.Qt.AlignCenter)
    font = item.font()
    font.setPointSizeF(13.0)
    font.setBold(True)
    item.setFont(font)
    item.setForeground(QtGui.QColor("#f0c96a"))
    return item


# ---------------------------------------------------------------- 界面


def run_gui(monitor: Monitor, autostart: bool = False) -> int:
    from PySide6 import QtCore, QtGui, QtWidgets

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(paths.APP_NAME)
    app.setOrganizationName(paths.APP_NAME)
    apply_theme(app)

    # 图标是只读资源，打包后跟着 exe 走
    icon_path = os.path.join(paths.resource_root(), "assets", "d4tracker.ico")
    icon = QtGui.QIcon(icon_path) if os.path.exists(icon_path) else QtGui.QIcon()
    if not icon.isNull():
        app.setWindowIcon(icon)

    # 界面设置存数据目录下的 ui.ini，不进注册表：便携、能看见、删掉就重置。
    # （存注册表的版本出过一次事：布局改版后窗口尺寸被旧值钉死，用户还没法自己清。）
    settings = QtCore.QSettings(os.path.join(paths.data_dir(), "ui.ini"),
                                QtCore.QSettings.IniFormat)

    win = QtWidgets.QWidget()
    win.setObjectName("root")
    win.setWindowTitle("暗黑4 副本计数器")
    if not icon.isNull():
        win.setWindowIcon(icon)
    win.setMinimumSize(960, 880)

    root = QtWidgets.QVBoxLayout(win)
    root.setContentsMargins(14, 14, 14, 12)
    root.setSpacing(10)

    # ============================ 顶部：状态 + 工具条 ============================
    head = QtWidgets.QHBoxLayout()
    head.setSpacing(12)

    lbl_state = QtWidgets.QLabel("已停止")
    lbl_state.setObjectName("stateOff")
    lbl_sub = QtWidgets.QLabel("")
    lbl_sub.setObjectName("sub")
    head.addWidget(lbl_state)
    head.addWidget(lbl_sub, 1)

    def tool_button(text: str, tip: str):
        b = QtWidgets.QPushButton(text)
        b.setToolTip(tip)
        head.addWidget(b)
        return b

    btn_start = tool_button("开始", "开始抓帧识别")
    btn_stop = tool_button("停止", "停止识别（不影响已记录的计数）")
    btn_stop.setEnabled(False)
    btn_undo = tool_button("撤销上一次", "删掉最新的一条自动记录")
    btn_open = tool_button("数据目录", "打开存放 counts.db 的目录")
    btn_about = tool_button("关于", "工作方式与合规说明")
    root.addLayout(head)

    # ============================ 中部：可拖拽的三段 ============================
    splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
    splitter.setChildrenCollapsible(False)
    splitter.setHandleWidth(6)
    root.addWidget(splitter, 1)

    def make_table(headers, selectable=False):
        t = QtWidgets.QTableWidget(0, len(headers))
        t.setHorizontalHeaderLabels(headers)
        t.verticalHeader().setVisible(False)
        t.verticalHeader().setDefaultSectionSize(34)
        t.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        t.setAlternatingRowColors(True)
        t.setShowGrid(False)
        if selectable:
            t.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
            t.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        else:
            t.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)

        hh = t.horizontalHeader()
        hh.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        for c in range(1, len(headers) - 1):
            hh.setSectionResizeMode(c, QtWidgets.QHeaderView.ResizeToContents)
        # 最后一列放 −/+ 按钮。ResizeToContents **不会把 setCellWidget 的宽度算进去**，
        # 列宽只按表头文字算，结果按钮被挤出去（实测 + 号直接超出表格右边缘）。
        # 所以这一列给固定宽度。
        hh.setSectionResizeMode(len(headers) - 1, QtWidgets.QHeaderView.Fixed)
        t.setColumnWidth(len(headers) - 1, 112)
        hh.setHighlightSections(False)
        return t

    def delta_cell(on_minus, on_plus):
        """每行两个按钮：−1 撤销最近一次，+1 手动补记。放一起是为了防误触。"""
        w = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        bm = QtWidgets.QPushButton("\u2212")
        bm.setObjectName("minus")
        bm.setFixedSize(40, 28)
        bm.setToolTip("撤销该项目最近一次计数（−1）")
        bp = QtWidgets.QPushButton("+")
        bp.setObjectName("plus")
        bp.setFixedSize(40, 28)
        bp.setToolTip("手动补记一次（+1）")
        bm.clicked.connect(on_minus)
        bp.clicked.connect(on_plus)
        lay.addWidget(bm)
        lay.addWidget(bp)
        return w

    # ---- 第 1 段：活动计数 + 自定义 ----
    pane_counts = QtWidgets.QWidget()
    pc = QtWidgets.QVBoxLayout(pane_counts)
    pc.setContentsMargins(0, 0, 0, 0)
    pc.setSpacing(8)

    g_counts = QtWidgets.QGroupBox("活动计数（自动识别）")
    gc = QtWidgets.QVBoxLayout(g_counts)
    t_counts = make_table(["活动", "今日", "总计", "手动"])
    # 最小高度按行数算出来，否则默认布局会把表格压到只露一行半，还得滚动才能看全
    t_counts.setMinimumHeight(len(kinds.TARGETS) * 34 + 40)
    gc.addWidget(t_counts)
    pc.addWidget(g_counts, 3)

    g_custom = QtWidgets.QGroupBox("自定义（不自动抓取，自己填）")
    gu = QtWidgets.QVBoxLayout(g_custom)

    crow = QtWidgets.QHBoxLayout()
    btn_add = QtWidgets.QPushButton("新增条目")
    btn_ren = QtWidgets.QPushButton("重命名")
    btn_del = QtWidgets.QPushButton("删除")
    for b in (btn_add, btn_ren, btn_del):
        crow.addWidget(b)
    hint = QtWidgets.QLabel("双击「今日」格可直接填数字；改名不会丢历史计数。")
    hint.setObjectName("hint")
    crow.addWidget(hint, 1)
    gu.addLayout(crow)

    t_custom = make_table(["条目", "今日", "总计", "手动"], selectable=True)
    t_custom.setMinimumHeight(3 * 34 + 40)
    gu.addWidget(t_custom)
    pc.addWidget(g_custom, 2)
    splitter.addWidget(pane_counts)

    # ---- 第 2 段：识别状态 ----
    g_watch = QtWidgets.QGroupBox("识别状态")
    gw = QtWidgets.QVBoxLayout(g_watch)

    prev_row = QtWidgets.QHBoxLayout()
    lbl_preview = QtWidgets.QLabel("横幅区域预览")
    lbl_preview.setObjectName("preview")
    lbl_preview.setFixedSize(430, 84)
    lbl_preview.setAlignment(QtCore.Qt.AlignCenter)
    prev_row.addWidget(lbl_preview)

    lbl_score = None          # 分数改用 2 列网格，见下方 score_grid
    score_labels: dict[str, tuple] = {}
    score_grid = QtWidgets.QGridLayout()
    score_grid.setHorizontalSpacing(18)
    score_grid.setVerticalSpacing(2)
    for i, (kind, label, _ready, _note) in enumerate(kinds.TARGETS):
        lab = QtWidgets.QLabel("")
        lab.setObjectName("score")
        lab.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        score_grid.addWidget(lab, i % 3, i // 3)
        score_labels[kind] = (lab, label)
    prev_row.addLayout(score_grid, 1)
    gw.addLayout(prev_row)

    vgrid = QtWidgets.QGridLayout()
    vgrid.setHorizontalSpacing(14)
    vgrid.setVerticalSpacing(4)
    value_labels: dict[str, QtWidgets.QLabel] = {}
    for vrow, slot in enumerate(monitor.values.slots):
        name_lab = QtWidgets.QLabel(slot.label)
        name_lab.setObjectName("vlabel")
        val_lab = QtWidgets.QLabel("—")
        val_lab.setObjectName("vvalue")
        vgrid.addWidget(name_lab, vrow, 0)
        vgrid.addWidget(val_lab, vrow, 1)
        value_labels[slot.key] = val_lab
    btn_reset_values = QtWidgets.QPushButton("清空累计")
    btn_reset_values.setToolTip("只清空数值记录，不动活动计数")
    vgrid.addWidget(btn_reset_values, 0, 2, max(1, len(monitor.values.slots)), 1)
    vgrid.setColumnStretch(1, 1)
    gw.addLayout(vgrid)
    splitter.addWidget(g_watch)

    # ---- 第 3 段：最近事件 + 日志 ----
    g_log = QtWidgets.QGroupBox("最近事件 / 运行日志")
    gl = QtWidgets.QVBoxLayout(g_log)
    log_row = QtWidgets.QHBoxLayout()

    list_recent = QtWidgets.QListWidget()
    log_row.addWidget(list_recent, 2)

    log_box = QtWidgets.QTextEdit()
    log_box.setReadOnly(True)
    log_box.setFont(QtGui.QFont("Consolas", 9))
    log_box.setLineWrapMode(QtWidgets.QTextEdit.NoWrap)
    log_row.addWidget(log_box, 3)
    gl.addLayout(log_row)
    splitter.addWidget(g_log)

    splitter.setSizes([560, 200, 240])
    if settings.value("window/splitter"):
        splitter.restoreState(settings.value("window/splitter"))

    lbl_warn = QtWidgets.QLabel("")
    lbl_warn.setObjectName("hint")
    lbl_warn.setWordWrap(True)
    root.addWidget(lbl_warn)

    # ============================ 行为 ============================

    def log_line(text: str, level: str = "info") -> None:
        colour = {"ok": "#4fc97a", "warn": "#e0a24a", "error": "#e0645e"}.get(level, "#b9b9c8")
        log_box.append(f'<span style="color:{colour}">{text}</span>')
        bar = log_box.verticalScrollBar()
        bar.setValue(bar.maximum())

    def manual(kind: str, label, delta: int) -> None:
        """界面上的 +1 / −1。计数是数出来的，−1 就是删掉最新那条记录。"""
        if delta > 0:
            monitor.store.record(kind, source="manual", label=label)
        else:
            monitor.store.remove_last(kind, label)
        refresh()

    def build_counts_rows() -> None:
        t_counts.setRowCount(len(kinds.TARGETS))
        for r, (kind, label, ready, _note) in enumerate(kinds.TARGETS):
            name = QtWidgets.QTableWidgetItem(label if ready else f"{label}（待标定）")
            name.setData(QtCore.Qt.UserRole, kind)
            name.setData(QtCore.Qt.UserRole + 1, None)
            if not ready:
                name.setForeground(QtGui.QColor("#7a7a88"))
            t_counts.setItem(r, 0, name)
            t_counts.setItem(r, 1, _num_item(0))
            t_counts.setItem(r, 2, _num_item(0))
            t_counts.setCellWidget(r, 3, delta_cell(
                (lambda _=False, k=kind: manual(k, None, -1)),
                (lambda _=False, k=kind: manual(k, None, +1))))

    def rebuild_custom() -> None:
        names = monitor.store.custom_items()
        if not names:
            t_custom.setRowCount(1)
            ph = QtWidgets.QTableWidgetItem("（还没有自定义条目，点上面「新增条目」）")
            ph.setForeground(QtGui.QColor("#6a6a78"))
            t_custom.setItem(0, 0, ph)
            for c in (1, 2, 3):
                t_custom.setItem(0, c, QtWidgets.QTableWidgetItem(""))
            return

        t_custom.setRowCount(len(names))
        for r, name in enumerate(names):
            it = QtWidgets.QTableWidgetItem(name)
            it.setData(QtCore.Qt.UserRole, kinds.CUSTOM_KIND)
            it.setData(QtCore.Qt.UserRole + 1, name)
            t_custom.setItem(r, 0, it)
            t_custom.setItem(r, 1, _num_item(0))
            t_custom.setItem(r, 2, _num_item(0))
            t_custom.setCellWidget(r, 3, delta_cell(
                (lambda _=False, n=name: manual(kinds.CUSTOM_KIND, n, -1)),
                (lambda _=False, n=name: manual(kinds.CUSTOM_KIND, n, +1))))

    def refresh() -> None:
        snap = monitor.snapshot()
        lbl_state.setText("● 监视中" if snap["running"] else "● 已停止")
        lbl_state.setObjectName("stateOn" if snap["running"] else "stateOff")
        lbl_state.style().unpolish(lbl_state)
        lbl_state.style().polish(lbl_state)

        lbl_sub.setText(
            f"游戏窗口 {'已找到' if snap['game_found'] else '未找到'}　·　"
            f"已抓帧 {snap['frames_seen']}（跳过 {snap['frames_skipped']}）　·　"
            f"单帧 {snap['capture_ms']:.0f} ms　·　"
            f"横幅在显：{', '.join(snap['active']) if snap['active'] else '—'}")

        preview = snap["preview"]
        if preview is not None and preview.size:
            rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            img = QtGui.QImage(rgb.data, w, h, rgb.strides[0], QtGui.QImage.Format_RGB888)
            lbl_preview.setPixmap(QtGui.QPixmap.fromImage(img).scaled(
                lbl_preview.size(), QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation))
        else:
            lbl_preview.setText("横幅区域预览")

        for kind, (lab, label) in score_labels.items():
            score = snap["scores"].get(kind, 0.0)
            best = snap.get("nearmiss", {}).get(kind, 0.0)
            mark = "***" if score >= detect.MATCH_THRESHOLD else "   "
            # 没到阈值时把「历史最高」也摆出来：0.000 和「最高 0.48」是完全不同的信息，
            # 前者是没出现，后者是位置/文字对不上
            hint = f"   最高 {best:.3f}" if (best > 0 and score < detect.MATCH_THRESHOLD) else ""
            lab.setText(f"{mark} {label}  {score:.3f}{hint}")

        today = monitor.store.counts_by_kind()
        total = monitor.store.counts_by_kind("all")
        for r in range(t_counts.rowCount()):
            kind = t_counts.item(r, 0).data(QtCore.Qt.UserRole)
            t_counts.item(r, 1).setText(str(today.get(kind, 0)))
            t_counts.item(r, 2).setText(str(total.get(kind, 0)))

        names = monitor.store.custom_items()
        if t_custom.rowCount() != max(1, len(names)):
            rebuild_custom()
        if names:
            ctoday = monitor.store.custom_counts()
            ctotal = monitor.store.custom_counts("all")
            for r, name in enumerate(names):
                if t_custom.item(r, 1):
                    t_custom.item(r, 1).setText(str(ctoday.get(name, 0)))
                if t_custom.item(r, 2):
                    t_custom.item(r, 2).setText(str(ctotal.get(name, 0)))

        for key, lab in value_labels.items():
            summary = monitor.store.readings_summary(key)
            cur = snap["values"].get(key)
            if cur is None:
                # 实时读不到（不在狂潮里，或刚启动）时，至少把上一次的记录摆出来 ——
                # 只写"未读到"会让人以为数据丢了
                if summary["latest"] is not None:
                    lab.setText(f"最近记录 {summary['latest']}      "
                                f"今日峰值 {summary['max']}      "
                                f"今日累计获得 {summary['gained']}      （当前未读到）")
                else:
                    lab.setText("无记录（当前不在地狱狂潮？）")
                continue
            lab.setText(f"{cur}      今日峰值 {summary['max']}      "
                        f"今日累计获得 {summary['gained']}      "
                        f"置信 {snap['value_confidence'].get(key, 0.0):.2f}")

        list_recent.clear()
        for ev in monitor.store.recent(30):
            t = time.strftime("%H:%M:%S", time.localtime(ev["ts"]))
            tag = "" if ev["source"] == "auto" else "  [手动]"
            score = f"  分数 {ev['score']:.3f}" if ev["source"] == "auto" else ""
            list_recent.addItem(f"{t}   {ev['label']}{score}{tag}")

        errs = snap["errors"]
        lbl_warn.setText(("最近错误：" + " / ".join(errs)) if errs else "")

        # 接近阈值 / 命中但帧数不足：都写进日志并留图，这是事后唯一能查的现场
        for name, label, score, path, why in monitor.drain_nearmiss():
            where = f"，图 {os.path.relpath(path, paths.data_dir())}" if path else ""
            log_line(f"{label}：{why}{where}", "warn")

    # ---- 双击「今日」格直接填数字 ----
    def on_double(table, row: int, col: int) -> None:
        if col != 1 or not table.item(row, 0):
            return
        item = table.item(row, 0)
        kind = item.data(QtCore.Qt.UserRole)
        label = item.data(QtCore.Qt.UserRole + 1)
        if kind is None:
            return
        current = int(table.item(row, 1).text() or 0)
        value, ok = QtWidgets.QInputDialog.getInt(
            win, "直接填写今日计数",
            f"「{item.text()}」今天记几次？\n（多了会补记录，少了会删掉最新的几条）",
            current, 0, 9999, 1)
        if ok:
            monitor.store.set_count(kind, label, value)
            refresh()

    t_counts.cellDoubleClicked.connect(lambda r, c: on_double(t_counts, r, c))
    t_custom.cellDoubleClicked.connect(lambda r, c: on_double(t_custom, r, c))

    # ---- 自定义条目的增删改 ----
    def selected_custom():
        row = t_custom.currentRow()
        if row < 0 or not t_custom.item(row, 0):
            return None
        return t_custom.item(row, 0).data(QtCore.Qt.UserRole + 1)

    def do_add() -> None:
        name, ok = QtWidgets.QInputDialog.getText(win, "新增自定义条目", "名称：")
        if not ok or not name.strip():
            return
        if monitor.store.add_custom(name):
            log_line(f"新增自定义条目「{name.strip()}」", "ok")
        else:
            log_line(f"「{name.strip()}」已存在或名称为空", "warn")
        rebuild_custom()
        refresh()

    def do_rename() -> None:
        old = selected_custom()
        if not old:
            log_line("请先在自定义表里选中一行", "warn")
            return
        name, ok = QtWidgets.QInputDialog.getText(win, "重命名", "新名称：", text=old)
        if not ok or not name.strip():
            return
        if monitor.store.rename_custom(old, name):
            log_line(f"「{old}」已改名为「{name.strip()}」", "ok")
        else:
            log_line("改名失败：名称为空或与新名称重复", "warn")
        rebuild_custom()
        refresh()

    def confirm(title: str, text: str) -> bool:
        """统一走这个而不是 QMessageBox.question：要能给它刷深色标题栏。"""
        box = QtWidgets.QMessageBox(win)
        box.setIcon(QtWidgets.QMessageBox.Question)
        box.setWindowTitle(title)
        box.setText(text)
        box.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        box.setDefaultButton(QtWidgets.QMessageBox.No)
        if not icon.isNull():
            box.setWindowIcon(icon)
        apply_dark_title_bar(box)
        return box.exec() == QtWidgets.QMessageBox.Yes

    def do_del() -> None:
        name = selected_custom()
        if not name:
            log_line("请先在自定义表里选中一行", "warn")
            return
        if confirm("删除自定义条目",
                   f"删除「{name}」？\n连同它的历史计数一起删除，不可恢复。"):
            monitor.store.remove_custom(name)
            log_line(f"已删除自定义条目「{name}」", "ok")
            rebuild_custom()
            refresh()

    btn_add.clicked.connect(do_add)
    btn_ren.clicked.connect(do_rename)
    btn_del.clicked.connect(do_del)

    # ---- 工具条 ----
    def do_start() -> None:
        monitor.start()
        btn_start.setEnabled(False)
        btn_stop.setEnabled(True)
        log_line("已开始监视", "ok")
        refresh()

    def do_stop() -> None:
        monitor.stop()
        btn_start.setEnabled(True)
        btn_stop.setEnabled(False)
        log_line("已停止监视")
        refresh()

    def do_undo() -> None:
        monitor.store.undo_last()
        log_line("已撤销最新一条记录")
        refresh()

    btn_start.clicked.connect(do_start)
    btn_stop.clicked.connect(do_stop)
    btn_undo.clicked.connect(do_undo)
    btn_open.clicked.connect(lambda: os.startfile(paths.data_dir()))
    btn_reset_values.clicked.connect(
        lambda: (monitor.store.reset_readings(),
                 log_line("已清空数值累计", "ok"), refresh()))
    btn_about.clicked.connect(lambda: _show_about())

    def _show_about() -> None:
        box = QtWidgets.QMessageBox(win)
        box.setIcon(QtWidgets.QMessageBox.Information)
        box.setWindowTitle("关于 / 合规说明")
        box.setText(
            "暗黑4 副本计数器\n\n"
            "工作方式：只读屏幕像素 + 图像识别（模板匹配 / 数字识别）。\n"
            "不读取游戏内存、不注入 DLL、不装键盘钩子、不模拟输入。\n\n"
            "为什么这么做：暴雪 EULA 禁止的是 modify / automate / interfere 类软件"
            "（官方蓝贴点名过读内存的 TurboHUD4，可永久封号）。"
            "纯屏幕识别不碰游戏进程，是同类别工具长期采用的做法。\n\n"
            f"数据目录：{paths.data_dir()}\n"
            f"模板目录：{detect.TEMPLATE_DIR}")
        box.setStandardButtons(QtWidgets.QMessageBox.Ok)
        if not icon.isNull():
            box.setWindowIcon(icon)
        apply_dark_title_bar(box)
        box.exec()

    timer = QtCore.QTimer()
    timer.timeout.connect(refresh)
    timer.start(400)

    def on_close(_e) -> None:
        monitor.stop()
        try:
            settings.setValue("window/geometry", win.saveGeometry())
            settings.setValue("window/splitter", splitter.saveState())
        except Exception:                                # noqa: BLE001
            pass

    win.closeEvent = on_close

    build_counts_rows()
    rebuild_custom()
    refresh()
    log_line("就绪。点「开始」开始监视。", "ok")
    if autostart:
        do_start()

    saved = settings.value("window/geometry")
    if not (saved and win.restoreGeometry(saved)):
        win.resize(1160, 1010)

    win.show()
    apply_dark_title_bar(win)

    # 排查用：D4TRACKER_DEBUG_UI=1 时打印真实布局数据并自动退出
    if os.environ.get("D4TRACKER_DEBUG_UI"):
        def _dump() -> None:
            geo = app.primaryScreen().availableGeometry()
            print(f"[ui] screen={geo.width()}x{geo.height()} "
                  f"窗口={win.width()}x{win.height()} "
                  f"minHint={win.minimumSizeHint().width()}x{win.minimumSizeHint().height()}")
            print(f"[ui] splitter={splitter.sizes()}  "
                  f"活动表={t_counts.height()} 自定义表={t_custom.height()} "
                  f"预览={lbl_preview.height()}")
            app.quit()

        QtCore.QTimer.singleShot(1200, _dump)

    return app.exec()


# ---------------------------------------------------------------- 其它入口


def run_headless(monitor: Monitor) -> int:
    monitor.start()
    print("D4 副本计数器（控制台）")
    print(f"阈值={detect.MATCH_THRESHOLD} 确认帧数={detect.CONFIRM_FRAMES} 帧率={monitor.fps}")
    print("数值指标: " + "、".join(s.label for s in monitor.values.slots))
    print("Ctrl+C 结束\n")
    try:
        seen_changes = 0
        while True:
            time.sleep(0.5)
            for ev in monitor.drain_events():
                print(f"[{ev.when}] 计数 +1  {ev.label}  分数={ev.score:.3f} "
                      f"连续={ev.frames}帧")
            with monitor.lock:
                changes = monitor.value_changes[seen_changes:]
                seen_changes = len(monitor.value_changes)
            for key, value, ts in changes:
                label = monitor.values.labels().get(key, key)
                print(f"[{time.strftime('%H:%M:%S', time.localtime(ts))}] "
                      f"数值变化  {label} = {value}")
    except KeyboardInterrupt:
        print("\n收尾中 ...")
    finally:
        monitor.stop()
    return 0


def run_selftest(samples_root: str) -> int:
    """自检入口：把 stdout 换成同时落文件的 TEE。

    `--noconsole` 打包的 exe 里 `sys.stdout` 是 None，`print()` 会抛
    AttributeError。这里直接换掉 stdout，下面那些 print 就不用逐个改，
    而且结果总会落到 data/selftest.log，图形界面里也能查。

    只写文件会走向另一个极端：源码运行时控制台反而什么都看不到，所以要回显。
    """

    class _Tee:
        def __init__(self, path: str, echo=None) -> None:
            self.fh = open(path, "w", encoding="utf-8")
            self.echo = echo

        def write(self, text: str) -> int:
            self.fh.write(text)
            self.fh.flush()
            if self.echo is not None:
                try:
                    self.echo.write(text)
                    self.echo.flush()
                except Exception:                        # noqa: BLE001
                    pass
            return len(text)

        def flush(self) -> None:
            self.fh.flush()

        def isatty(self) -> bool:
            return False

    old = sys.stdout
    try:
        sys.stdout = _Tee(paths.log_path("selftest.log"), echo=old)
    except OSError:
        return _run_selftest(samples_root)

    try:
        return _run_selftest(samples_root)
    finally:
        sys.stdout = old
        try:
            print(f"自检报告已写入 {paths.log_path('selftest.log')}")
        except Exception:                                # noqa: BLE001
            pass


def _run_selftest(samples_root: str) -> int:
    """跑一遍「检测 -> 状态机 -> 落库」，确认全链路通。

    样本是可选的：采集的原始帧很占地方（1GB+），标定完之后通常会被清掉。
    没有样本时跳过回放部分，仍然把模板、存储 API、数值读取路径校验一遍 ——
    否则「清理样本」就等于「自检永远失败」，那就没人会再跑它了。
    """
    import tempfile

    det = detect.Detector()
    if not det.ready:
        print("没有模板，先跑 tools/train.py")
        return 1

    print(f"模板 {len(det.templates)} 个:")
    for t in det.templates:
        kind = f"搜索 strip={t.strip}" if t.search else f"ROI={t.roi}"
        print(f"  {t.label:14s} {kind}  来自 {t.source}")

    tmp = os.path.join(tempfile.gettempdir(), "d4tracker-selftest.db")
    if os.path.exists(tmp):
        os.remove(tmp)
    store = Store(tmp)
    mon = Monitor(store, det)

    ok = True
    have_samples = os.path.isdir(samples_root) and any(
        d for d in os.listdir(samples_root)
        if os.path.isdir(os.path.join(samples_root, d))
        and not d.startswith("_") and d != "crops")

    target = None
    if have_samples:
        for d in sorted(os.listdir(samples_root)):
            if "梦魇地下城-完成" in d and os.path.isdir(os.path.join(samples_root, d)):
                target = os.path.join(samples_root, d)
                break
        if target is None:
            print("\n样本目录里没有梦魇样本，跳过回放")
            have_samples = False

    if have_samples:
        files = sorted(f for f in os.listdir(target)
                       if f.startswith("f") and f.endswith(".jpg"))
        print(f"\n回放 {os.path.basename(target)}  共 {len(files)} 帧")

        # 时间戳要落在"今天"，否则今日计数查不到 —— 第一版这里写成 1000.0（1970 年），
        # 自检报"今日计数不对"，实际是测试自己造的假数据不对。
        base = time.time() - len(files) / 3.0
        for i, fn in enumerate(files):
            img = imageio.imread(os.path.join(target, fn))
            if img is None:
                continue
            mon.process_frame(img, now=base + i / 3.0)

        total = store.total("all")
        print(f"入库事件 {total} 条")
        for row in store.recent(5):
            print(f"  {row['label']}  分数={row['score']:.3f}  来源={row['source']}")
    else:
        print("\n样本已清理：跳过基于样本的回放，只校验模板与存储")
        total = 0

    # 存储 API 自检：界面依赖这些查询，别等打开界面才发现写错
    print()
    today = store.counts_by_kind()
    allc = store.counts_by_kind("all")
    print(f"今日计数 {today}")
    if have_samples:
        if allc.get("nightmare_complete") != 1:
            print("  !! counts_by_kind 不对")
            ok = False
        if today.get("nightmare_complete") != 1:
            print("  !! 今日计数不对")
            ok = False
        if len(store.days()) < 1:
            print("  !! days() 没返回数据")
            ok = False

    manual_id = store.record("dungeon_complete", source="manual")
    if store.counts_by_kind("all").get("dungeon_complete") != 1:
        print("  !! 手工补记没进统计")
        ok = False
    if not store.delete(manual_id) or store.counts_by_kind("all").get("dungeon_complete") != 0:
        print("  !! 删除没生效")
        ok = False

    # 未标定的活动也必须在清单里（界面要显示「待标定」而不是消失）
    for k in kinds.all_kinds():
        if k not in store.counts_by_kind("all"):
            print(f"  !! 计数表缺少 {k}")
            ok = False

    # 自定义条目：加 / 计数 / 改名不丢历史 / 直接设定数值 / −1 / 删除清账
    print()
    if store.add_custom("自测条目") and not store.add_custom("自测条目"):
        print("自定义条目：新增 + 重名拦截 正常")
    else:
        print("  !! 自定义条目新增/重名拦截异常")
        ok = False

    store.record(kinds.CUSTOM_KIND, source="manual", label="自测条目")
    store.record(kinds.CUSTOM_KIND, source="manual", label="自测条目")
    if store.custom_counts().get("自测条目") == 2:
        print("自定义条目：计数正常")
    else:
        print("  !! 自定义条目计数不对")
        ok = False

    store.rename_custom("自测条目", "改过名")
    if store.custom_counts().get("改过名") == 2:
        print("自定义条目：改名后历史计数保留")
    else:
        print("  !! 改名后历史计数丢了")
        ok = False

    store.set_count(kinds.CUSTOM_KIND, "改过名", 5)
    if store.custom_counts().get("改过名") == 5:
        print("自定义条目：直接设定数值（5）正常")
    else:
        print("  !! set_count 不对")
        ok = False

    store.remove_last(kinds.CUSTOM_KIND, "改过名")
    if store.custom_counts().get("改过名") == 4:
        print("自定义条目：−1 正常")
    else:
        print("  !! remove_last 不对")
        ok = False

    store.remove_custom("改过名")
    if not store.custom_items() and not store.custom_counts("all"):
        print("自定义条目：删除并清账正常")
    else:
        print("  !! 删除自定义条目没清干净")
        ok = False

    # 数值读取（地狱狂潮）：拿一组狂潮样本回放，验证投票与落库
    hell = None
    if have_samples:
        for d in sorted(os.listdir(samples_root)):
            if d.startswith("地狱狂潮") and os.path.isdir(os.path.join(samples_root, d)):
                hell = os.path.join(samples_root, d)
                break
    if hell:
        print(f"\n-- 数值读取自检：{os.path.basename(hell)} --")
        vmon = Monitor(store, det, value_interval=1)
        vfiles = sorted(f for f in os.listdir(hell)
                        if f.startswith("f") and f.endswith(".jpg"))
        for fn in vfiles:
            vimg = imageio.imread(os.path.join(hell, fn))
            if vimg is not None:
                vmon.read_values(vimg)
        print(f"确认读数 {vmon.latest_values}  置信 {vmon.value_confidence}")
        for key in vmon.latest_values:
            summary = store.readings_summary(key)
            print(f"  {vmon.values.labels().get(key, key)}: 最新={summary['latest']} "
                  f"峰值={summary['max']} 累计获得={summary['gained']} "
                  f"记录 {summary['samples']} 次变化")
        if not vmon.latest_values:
            print("  !! 没读到任何数值")
            ok = False
    else:
        # 没有样本也要确认数值通路能用：造一张纯黑图，读不出数字但不该抛异常
        print("\n-- 数值读取通路（无样本，仅确认不抛异常）--")
        blank = np.zeros((detect.REF_H, detect.REF_W, 3), dtype=np.uint8)
        try:
            vmon = Monitor(store, det, value_interval=1)
            vmon.read_values(blank)
            print(f"  空白帧读数 {vmon.latest_values}（应为空）")
        except Exception as exc:                        # noqa: BLE001
            print(f"  !! 数值读取抛异常: {type(exc).__name__}: {exc}")
            ok = False

    # ---- 状态机：迟滞（这是"同一段文案被记很多次"的根因，必须回归）----
    print("\n-- 状态机：迟滞与重复触发 --")
    ok = _check_counter(ok)

    store.close()
    if os.path.exists(tmp):
        os.remove(tmp)
    if have_samples:
        ok = ok and total == 1
    print("全链路自检: " + ("通过" if ok else "失败"))
    return 0 if ok else 1


def _check_counter(ok: bool) -> bool:
    """状态机的关键行为，不需要游戏也不需要样本。

    实战里出现过 90 秒记 3 次巢穴首领（分数几乎相同），根因就是释放判定也用触发
    阈值：文案常驻时分数一抖就"释放→再触发"。这里把迟滞的行为钉死。
    """
    def run(scores, release_threshold):
        c = EventCounter(threshold=detect.MATCH_THRESHOLD,
                         confirm_frames=detect.CONFIRM_FRAMES,
                         release_threshold=release_threshold)
        fired, t = [], 1000.0
        for s in scores:
            fired += c.update({"k": s}, now=t)
            t += 1.0 / 3.0
        return fired

    th = detect.MATCH_THRESHOLD

    # 常驻文案 + 周期性短暂掉到 RELEASE_THRESHOLD 之上：只该记 1 次
    flicker = ([0.8] * 6 + [RELEASE_THRESHOLD + 0.05] * 18) * 12
    n_new = len(run(flicker, RELEASE_THRESHOLD))
    n_old = len(run(flicker, th))
    print(f"  文案抖动 96 秒: 迟滞={n_new} 次   旧逻辑={n_old} 次（应为 {n_new} < {n_old}）")
    if n_new != 1 or n_old <= 1:
        print("  !! 迟滞没起作用")
        ok = False

    # 真的消失足够久再来 -> 必须还能记第二次
    gone = [0.8] * 6 + [0.1] * 75 + [0.8] * 6
    n2 = len(run(gone, RELEASE_THRESHOLD))
    print(f"  真消失 25 秒后再来: {n2} 次（应为 2）")
    if n2 != 2:
        print("  !! 正常重复被误吞")
        ok = False

    # 正常横幅：停留 4 秒后彻底消失 -> 1 次
    n3 = len(run([0.98] * 12 + [0.3] * 9 + [0.15] * 9, RELEASE_THRESHOLD))
    print(f"  正常横幅: {n3} 次（应为 1）")
    if n3 != 1:
        print("  !! 正常横幅计数异常")
        ok = False

    # 向后兼容：不传 release_threshold 时行为等同旧版
    if EventCounter(th).release_threshold != th:
        print("  !! 默认构造改变了旧行为")
        ok = False
    return ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="D4 副本计数器")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--fps", type=float, default=3.0)
    ap.add_argument("--ignore-focus", action="store_true",
                    help="不要求游戏在前台（调试用）")
    ap.add_argument("--autostart", action="store_true",
                    help="打开界面后立即开始监视（不用点「开始」）")
    ap.add_argument("--db", default=DB_PATH)
    args = ap.parse_args(argv)

    samples = os.path.join(ROOT, "samples")
    if args.selftest:
        return run_selftest(samples)

    det = detect.Detector()
    if not det.ready:
        print("没有模板，先跑 tools/train.py")
        return 1

    store = Store(args.db)
    mon = Monitor(store, det, fps=args.fps,
                  only_when_foreground=not args.ignore_focus)
    try:
        if args.headless:
            return run_headless(mon)
        return run_gui(mon, autostart=args.autostart)
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
