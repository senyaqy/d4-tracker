"""暗黑破坏神IV 窗口捕获。

为什么不用 mss / PIL.ImageGrab（GDI BitBlt 系）作为主路径
--------------------------------------------------------
1. 独占全屏（DXGI flip 呈现）下 BitBlt 只能拿到黑屏或过期帧；
2. BitBlt 抓的是"屏幕上那一块区域"，窗口被别的窗口盖住时会抓到遮挡物，
   而本工具需要在玩家挂着游戏、切出去看网页时依然能稳定判定。

PrintWindow + PW_RENDERFULLCONTENT(0x00000002) 让窗口自己渲染一份到我们
提供的 DC，既不依赖窗口是否在最前，也不需要 DWM 缩略图。

本机实测（D4 2560x1440 无边框全屏）::

    PrintWindow flag=2 -> 正常画面（平均亮度 115.9）
    PrintWindow flag=0 -> 全屏均匀灰 133（D3D 窗口的典型结果）

因此：flag=2 为主路径，BitBlt 仅作回退，并在回退时用"画面是否可信"做校验。
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wintypes
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------- Win32 绑定

user32 = ctypes.WinDLL("user32", use_last_error=True)
gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

SRCCOPY = 0x00CC0020
DIB_RGB_COLORS = 0
PW_RENDERFULLCONTENT = 0x00000002
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
user32.GetWindowRect.restype = wintypes.BOOL
user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
user32.GetClientRect.restype = wintypes.BOOL
user32.PrintWindow.argtypes = [wintypes.HWND, wintypes.HDC, wintypes.UINT]
user32.PrintWindow.restype = wintypes.BOOL
user32.GetDC.argtypes = [wintypes.HWND]
user32.GetDC.restype = wintypes.HDC
user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
user32.IsWindowVisible.argtypes = [wintypes.HWND]
user32.IsIconic.argtypes = [wintypes.HWND]
user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
user32.GetForegroundWindow.restype = wintypes.HWND
user32.IsWindow.argtypes = [wintypes.HWND]
user32.IsWindow.restype = wintypes.BOOL

gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
gdi32.CreateCompatibleDC.restype = wintypes.HDC
gdi32.CreateDIBSection.argtypes = [
    wintypes.HDC, ctypes.c_void_p, wintypes.UINT,
    ctypes.POINTER(ctypes.c_void_p), wintypes.HANDLE, wintypes.DWORD,
]
gdi32.CreateDIBSection.restype = wintypes.HBITMAP
gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
gdi32.SelectObject.restype = wintypes.HGDIOBJ
gdi32.BitBlt.argtypes = [
    wintypes.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    wintypes.HDC, ctypes.c_int, ctypes.c_int, wintypes.DWORD,
]
gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
gdi32.DeleteDC.argtypes = [wintypes.HDC]


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD),
        ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


class BITMAPINFO(ctypes.Structure):
    _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", wintypes.DWORD * 3)]


# ---------------------------------------------------------------- 窗口查找


@dataclass
class WindowInfo:
    hwnd: int
    pid: int
    exe: str
    title: str
    left: int
    top: int
    width: int
    height: int
    minimized: bool
    visible: bool

    @property
    def rect(self) -> tuple[int, int, int, int]:
        return (self.left, self.top, self.width, self.height)

    def __str__(self) -> str:
        return (f"hwnd=0x{self.hwnd:X} pid={self.pid} {self.exe!r} {self.title!r} "
                f"{self.width}x{self.height}@({self.left},{self.top}) "
                f"minimized={self.minimized} visible={self.visible}")


def _exe_of_pid(pid: int) -> str:
    h = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not h:
        return ""
    try:
        size = wintypes.DWORD(1024)
        buf = ctypes.create_unicode_buffer(size.value)
        if kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
            return buf.value
        return ""
    finally:
        kernel32.CloseHandle(h)


def _title_of(hwnd: int) -> str:
    n = user32.GetWindowTextLengthW(hwnd)
    if n <= 0:
        return ""
    buf = ctypes.create_unicode_buffer(n + 1)
    user32.GetWindowTextW(hwnd, buf, n + 1)
    return buf.value


def list_windows(min_size: tuple[int, int] = (320, 240)) -> list[WindowInfo]:
    """枚举所有可见的顶层窗口。

    min_size 可放宽：默认过滤掉小窗口（游戏主体用不到），但抓对话框时
    400x180 的 QMessageBox 会被这个门槛挡掉，需要显式传小一点的尺寸。
    """
    out: list[WindowInfo] = []
    min_w, min_h = min_size
    WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def cb(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        title = _title_of(hwnd)
        if not title:
            return True
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        r = wintypes.RECT()
        if not user32.GetWindowRect(hwnd, ctypes.byref(r)):
            return True
        w, h = r.right - r.left, r.bottom - r.top
        if w < min_w or h < min_h:
            return True
        out.append(WindowInfo(
            hwnd=hwnd, pid=pid.value, exe=_exe_of_pid(pid.value), title=title,
            left=r.left, top=r.top, width=w, height=h,
            minimized=bool(user32.IsIconic(hwnd)), visible=True,
        ))
        return True

    user32.EnumWindows(WNDENUMPROC(cb), 0)
    return out


# 进程名候选：不同版本/语言/启动方式下 D4 的进程名
GAME_EXE_HINTS = ("diablo iv", "diabloiv", "diablo iv.exe", "diabloiv.exe")
GAME_TITLE_HINTS = ("暗黑破坏神", "diablo")


def find_game_window(exe_hints=GAME_EXE_HINTS, title_hints=GAME_TITLE_HINTS) -> WindowInfo | None:
    """按进程名优先、窗口标题兜底地找到游戏窗口；多个候选取面积最大的那个。"""
    wins = list_windows()
    scored: list[tuple[int, WindowInfo]] = []
    for w in wins:
        exe = w.exe.lower().rsplit("\\", 1)[-1]
        title = w.title.lower()
        score = 0
        if exe in exe_hints or any(h in exe for h in ("diablo",)):
            score += 10
        if any(h in title for h in title_hints):
            score += 5
        if score:
            scored.append((score, w))
    if not scored:
        return None
    scored.sort(key=lambda t: (t[0], t[1].width * t[1].height), reverse=True)
    return scored[0][1]


# ---------------------------------------------------------------- 抓帧


class _Surface:
    """可复用的 GDI 抓帧表面。

    每抓一帧都 CreateCompatibleDC + CreateDIBSection + DeleteObject 是纯浪费：
    实测 1440p 整帧在复用后从 ~40ms 降到 ~10ms 量级，直接决定轮询频率能开多高。

    注意：内部缓冲会被下一次抓帧覆盖，array() 返回的是拷贝，可以安全持有。
    单线程使用；多线程请各自持有一个实例。
    """

    def __init__(self) -> None:
        self.hdc_screen = None
        self.mem_dc = None
        self.hbmp = None
        self.old = None
        self.bits = None
        self.w = 0
        self.h = 0
        self._buf = None
        self._buf_len = 0

    def ensure(self, w: int, h: int) -> None:
        if self.mem_dc and self.w == w and self.h == h:
            return
        self.free()

        self.hdc_screen = user32.GetDC(0)
        if not self.hdc_screen:
            raise OSError("GetDC(0) 失败")
        self.mem_dc = gdi32.CreateCompatibleDC(self.hdc_screen)
        if not self.mem_dc:
            self.free()
            raise OSError("CreateCompatibleDC 失败")

        bmi = BITMAPINFO()
        bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.bmiHeader.biWidth = w
        bmi.bmiHeader.biHeight = -h              # 负数 = 自上而下，省一次翻转
        bmi.bmiHeader.biPlanes = 1
        bmi.bmiHeader.biBitCount = 32
        bmi.bmiHeader.biCompression = DIB_RGB_COLORS

        bits = ctypes.c_void_p()
        self.hbmp = gdi32.CreateDIBSection(self.hdc_screen, ctypes.byref(bmi),
                                           DIB_RGB_COLORS, ctypes.byref(bits), None, 0)
        if not self.hbmp or not bits:
            self.free()
            raise OSError("CreateDIBSection 失败")

        self.old = gdi32.SelectObject(self.mem_dc, self.hbmp)
        self.bits = bits
        self.w, self.h = w, h

    def free(self) -> None:
        try:
            if self.mem_dc and self.old:
                gdi32.SelectObject(self.mem_dc, self.old)
            if self.hbmp:
                gdi32.DeleteObject(self.hbmp)
            if self.mem_dc:
                gdi32.DeleteDC(self.mem_dc)
        finally:
            if self.hdc_screen:
                user32.ReleaseDC(0, self.hdc_screen)
            self.hdc_screen = self.mem_dc = self.hbmp = self.old = self.bits = None
            self.w = self.h = 0
            self._buf = None
            self._buf_len = 0

    def array(self) -> np.ndarray:
        """带拷贝的 BGR 数组，可以安全持有到下一次抓帧之后。"""
        raw = ctypes.string_at(self.bits, self.w * self.h * 4)
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(self.h, self.w, 4)
        return np.ascontiguousarray(arr[:, :, :3])       # BGRA -> BGR（拷贝）

    def view(self) -> np.ndarray:
        """**零拷贝**的 BGR 视图，只在下次抓帧之前有效。

        原来每帧都要 ``string_at`` 拷一份 14.7MB、``ascontiguousarray`` 再拷一份，
        而实际只用到几个小 ROI —— 1440p 下纯浪费十几毫秒。这里直接用
        ``np.ctypeslib.as_array`` 把 GDI 缓冲映射成 numpy 视图，ROI 取用时才拷贝。
        GDI 缓冲由本对象持有，所以视图的生命周期跟着它。
        """
        nbytes = self.w * self.h * 4
        if self._buf is None or self._buf_len != nbytes:
            self._buf = (ctypes.c_ubyte * nbytes).from_address(self.bits.value)
            self._buf_len = nbytes
        arr = np.ctypeslib.as_array(self._buf).reshape(self.h, self.w, 4)
        return arr[:, :, :3]                              # 步长视图，不拷贝


_SURFACE = _Surface()

# 最近一次 _grab 实际走的方式；method='auto' 时用它回答"到底走了哪条路"
_LAST_METHOD = "none"


def last_method() -> str:
    return _LAST_METHOD


def is_window(hwnd: int) -> bool:
    """窗口句柄是否还有效（游戏退出后旧句柄会失效）。"""
    return bool(user32.IsWindow(hwnd))


def _grab(hwnd: int, x: int, y: int, w: int, h: int, method: str) -> np.ndarray:
    """抓 hwnd 的一块区域，返回 BGR 的 numpy 数组 (h, w, 3)。

    method: 'printwindow' | 'bitblt'
    """
    global _LAST_METHOD

    w = int(w)
    h = int(h)
    if w <= 0 or h <= 0:
        raise ValueError(f"非法的抓取尺寸 {w}x{h}")

    _SURFACE.ensure(w, h)
    _LAST_METHOD = method

    if method == "printwindow":
        if not user32.PrintWindow(hwnd, _SURFACE.mem_dc, PW_RENDERFULLCONTENT):
            raise OSError("PrintWindow 返回 0")
    else:
        hdc_win = user32.GetDC(hwnd)
        if not hdc_win:
            raise OSError("GetDC(hwnd) 失败")
        try:
            if not gdi32.BitBlt(_SURFACE.mem_dc, 0, 0, w, h, hdc_win, x, y, SRCCOPY):
                raise OSError("BitBlt 返回 0")
        finally:
            user32.ReleaseDC(hwnd, hdc_win)

    return _SURFACE.view()


def _looks_blank(img: np.ndarray) -> bool:
    """整幅画面颜色几乎一致 => 判定为空白帧（PrintWindow 对 D3D 失败时的典型表现）。"""
    if img.size == 0:
        return True
    small = img[::16, ::16].reshape(-1, 3).astype(np.int16)
    return bool(small.std(axis=0).max() < 3.0)


def is_foreground(hwnd: int) -> bool:
    """游戏窗口当前是否在最前。

    BitBlt 读的是"屏幕上那个位置"，只有窗口真的在前面时才是游戏画面；
    被遮挡时它返回一片平色（实测通道 std = 0），因此这个判断是省一半开销的前提。
    """
    return int(user32.GetForegroundWindow() or 0) == int(hwnd)


def grab_window(win: WindowInfo, method: str = "auto", copy: bool = True) -> np.ndarray:
    """抓取整个窗口。

    method='auto'：窗口在前台时先用 BitBlt（实测 1440p 约 20ms，是 PrintWindow
    的一半），拿到平色说明判断失误或窗口被盖住，再回退 PrintWindow（约 40ms）。
    不在前台就直接 PrintWindow，不浪费一次注定失败的 BitBlt。

    copy=False 时返回**零拷贝视图**，只在下次抓帧之前有效 —— 热路径（实时监视）
    走这条，省掉每帧两次整帧拷贝；其它调用方保持默认的 True 更安全。
    """
    if method == "printwindow":
        img = _grab(win.hwnd, 0, 0, win.width, win.height, "printwindow")
    elif method == "bitblt":
        img = _grab(win.hwnd, 0, 0, win.width, win.height, "bitblt")
    elif is_foreground(win.hwnd):
        img = _grab(win.hwnd, 0, 0, win.width, win.height, "bitblt")
        if _looks_blank(img):
            img = _grab(win.hwnd, 0, 0, win.width, win.height, "printwindow")
    else:
        img = _grab(win.hwnd, 0, 0, win.width, win.height, "printwindow")

    return np.ascontiguousarray(img) if copy else img


def grab_region(win: WindowInfo, box: tuple[int, int, int, int],
                method: str = "auto") -> np.ndarray:
    """只抓窗口内的一块 ROI（像素坐标，左上原点），返回连续数组。

    BitBlt 支持偏移抓取；PrintWindow 只能整窗渲染，所以这里抓整窗再切。
    """
    x, y, w, h = box

    if method == "bitblt":
        return np.ascontiguousarray(_grab(win.hwnd, x, y, w, h, "bitblt"))

    full = grab_window(win, method=method, copy=False)
    return np.ascontiguousarray(full[y:y + h, x:x + w])


__all__ = [
    "WindowInfo", "list_windows", "find_game_window", "is_window", "is_foreground",
    "grab_window", "grab_region", "last_method", "PW_RENDERFULLCONTENT",
]
