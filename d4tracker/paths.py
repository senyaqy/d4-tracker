"""运行路径：源码运行 与 打包成 exe 两种形态下的目录解析。

打包后必须区分两类目录，否则会踩两个坑
--------------------------------------
1. **PyInstaller 单文件模式**启动时把整个包解压到 `sys._MEIPASS` 指向的临时目录，
   进程退出就删掉。把 `counts.db` 放在那里等于每次启动都从零开始，而且写不进去。
   所以**可写数据跟着 exe 走**（exe 所在目录），exe 在只读位置时再退回
   `%LOCALAPPDATA%\\D4Tracker`。
2. **只读资源**（`templates/`）则应该跟着包走，这样 exe 是自包含的，
   拷到哪台机器都能用。

源码运行时两者都是项目根目录，行为和以前一致。
"""

from __future__ import annotations

import os
import sys

FROZEN = bool(getattr(sys, "frozen", False))
APP_NAME = "D4Tracker"


def resource_root() -> str:
    """只读资源根目录（`templates/` 等）。"""
    if FROZEN:
        return getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(sys.executable)))
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def app_root() -> str:
    """可写数据的首选根目录：打包后是 exe 所在目录，源码运行时是项目根。"""
    if FROZEN:
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _writable(path: str) -> bool:
    try:
        os.makedirs(path, exist_ok=True)
        probe = os.path.join(path, ".write-probe")
        with open(probe, "w", encoding="utf-8") as fh:
            fh.write("")
        os.remove(probe)
        return True
    except OSError:
        return False


def data_dir() -> str:
    """可写数据目录。优先 exe 旁边（便携），不可写则退回 LOCALAPPDATA。"""
    preferred = os.path.join(app_root(), "data")
    if _writable(preferred):
        return preferred
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    fallback = os.path.join(base, APP_NAME, "data")
    os.makedirs(fallback, exist_ok=True)
    return fallback


def db_path() -> str:
    return os.path.join(data_dir(), "counts.db")


def log_path(name: str = "d4tracker.log") -> str:
    return os.path.join(data_dir(), name)


def has_console() -> bool:
    """打包成 --noconsole 时 sys.stdout 是 None，print 会直接抛异常。"""
    return sys.stdout is not None and hasattr(sys.stdout, "write")


def emit(message: str, log_name: str = "d4tracker.log") -> None:
    """既能进控制台也能落文件 —— 无控制台的 exe 里就只剩文件这条路。"""
    if has_console():
        try:
            print(message)
        except Exception:                                # noqa: BLE001
            pass
    try:
        with open(log_path(log_name), "a", encoding="utf-8") as fh:
            fh.write(message + "\n")
    except OSError:
        pass


__all__ = ["FROZEN", "APP_NAME", "resource_root", "app_root", "data_dir",
           "db_path", "log_path", "has_console", "emit"]
