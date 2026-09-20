"""全局热键（Win32 RegisterHotKey）。

为什么不用 keyboard / pynput
----------------------------
它们靠 ``SetWindowsHookEx`` 装全局低级键盘钩子，会拦下系统里所有按键。一个常驻
在游戏旁边的工具没必要这么做，而且游戏反作弊对输入钩子敏感。``RegisterHotKey``
只是向系统登记几个组合键，由系统在命中时给我们投递 ``WM_HOTKEY``：不注入 DLL、
不装钩子、不模拟输入，尺度上安全得多。

为什么必须全局
--------------
游戏独占键盘焦点，任何"窗口内快捷键"在游戏里都收不到。

实现要点
--------
``RegisterHotKey(None, id, ...)`` 把热键登记到**调用线程**的消息队列上，所以
登记、消息循环、注销必须在同一个线程里完成 —— 这就是本模块自建线程的原因。
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wintypes
import queue
import threading

user32 = ctypes.WinDLL("user32", use_last_error=True)

MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
MOD_WIN = 0x0008
MOD_NOREPEAT = 0x4000          # 按住不放时只触发一次

WM_HOTKEY = 0x0312
WM_QUIT = 0x0012

user32.RegisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int, wintypes.UINT, wintypes.UINT]
user32.RegisterHotKey.restype = wintypes.BOOL
user32.UnregisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int]
user32.GetMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND,
                               wintypes.UINT, wintypes.UINT]
user32.PostThreadMessageW.argtypes = [wintypes.DWORD, wintypes.UINT,
                                      wintypes.WPARAM, wintypes.LPARAM]

_MODS = {
    "ctrl": MOD_CONTROL, "control": MOD_CONTROL,
    "alt": MOD_ALT,
    "shift": MOD_SHIFT,
    "win": MOD_WIN, "super": MOD_WIN, "meta": MOD_WIN,
}

_NAMED_KEYS = {
    "space": 0x20, "esc": 0x1B, "escape": 0x1B, "tab": 0x09,
    "enter": 0x0D, "return": 0x0D, "backspace": 0x08,
    "insert": 0x2D, "delete": 0x2E, "del": 0x2E,
    "home": 0x24, "end": 0x23, "pageup": 0x21, "pagedown": 0x22,
    "up": 0x26, "down": 0x28, "left": 0x25, "right": 0x27,
    "num0": 0x60, "num1": 0x61, "num2": 0x62, "num3": 0x63, "num4": 0x64,
    "num5": 0x65, "num6": 0x66, "num7": 0x67, "num8": 0x68, "num9": 0x69,
}
for _i in range(1, 25):
    _NAMED_KEYS[f"f{_i}"] = 0x6F + _i


def parse_hotkey(spec: str) -> tuple[int, int]:
    """把 ``"Ctrl+Alt+3"`` 解析成 ``(modifiers, virtual_key)``。

    正则做不到的事这里也不做：不做模糊匹配，无法识别就抛 ValueError，
    免得用户以为登记成功了、实际什么都没发生。
    """
    parts = [p.strip().lower() for p in spec.split("+") if p.strip()]
    if not parts:
        raise ValueError(f"空的热键定义: {spec!r}")

    mods = 0
    key = None
    for p in parts:
        if p in _MODS:
            mods |= _MODS[p]
        elif key is None:
            key = p
        else:
            raise ValueError(f"热键里有多个主键: {spec!r}")

    if key is None:
        raise ValueError(f"热键缺少主键: {spec!r}")

    if key in _NAMED_KEYS:
        vk = _NAMED_KEYS[key]
    elif len(key) == 1 and key.isdigit():
        vk = 0x30 + int(key)
    elif len(key) == 1 and "a" <= key <= "z":
        vk = ord(key.upper())
    else:
        raise ValueError(f"无法识别的按键 {key!r}（定义: {spec!r}）")

    return mods, vk


class HotkeyListener:
    """在独立线程里登记一组全局热键，命中时把标签放进 ``queue``。

    ``failed`` 保存登记失败的热键（通常是和别的程序冲突）—— 调用方应该把
    它显示出来，否则用户按半天没反应会以为工具坏了。
    """

    def __init__(self, mapping: dict[str, str]) -> None:
        self.mapping = dict(mapping)
        self.queue: queue.Queue[str] = queue.Queue()
        self.failed: list[tuple[str, str]] = []
        self.registered: list[tuple[str, str]] = []
        self._thread: threading.Thread | None = None
        self._thread_id: int | None = None
        self._ready = threading.Event()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="hotkeys", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5.0)

    def stop(self) -> None:
        if self._thread_id:
            user32.PostThreadMessageW(self._thread_id, WM_QUIT, 0, 0)
        if self._thread:
            self._thread.join(timeout=3.0)

    def _run(self) -> None:
        self._thread_id = ctypes.windll.kernel32.GetCurrentThreadId()
        id_to_label: dict[int, str] = {}

        hid = 1
        for spec, label in self.mapping.items():
            try:
                mods, vk = parse_hotkey(spec)
            except ValueError as exc:
                self.failed.append((spec, str(exc)))
                continue
            if user32.RegisterHotKey(None, hid, mods | MOD_NOREPEAT, vk):
                id_to_label[hid] = label
                self.registered.append((spec, label))
                hid += 1
            else:
                err = ctypes.get_last_error()
                self.failed.append((spec, f"RegisterHotKey 失败 (WinError {err})，多半被别的程序占用"))

        self._ready.set()

        msg = wintypes.MSG()
        try:
            while True:
                ret = user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
                if ret in (0, -1):
                    break
                if msg.message == WM_HOTKEY:
                    label = id_to_label.get(int(msg.wParam))
                    if label is not None:
                        self.queue.put(label)
                user32.TranslateMessage(ctypes.byref(msg))
                user32.DispatchMessageW(ctypes.byref(msg))
        finally:
            for hid_ in id_to_label:
                user32.UnregisterHotKey(None, hid_)


__all__ = ["HotkeyListener", "parse_hotkey", "MOD_CONTROL", "MOD_ALT", "MOD_SHIFT", "MOD_WIN"]
