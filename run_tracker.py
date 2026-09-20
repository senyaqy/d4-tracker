"""打包入口 —— 也是双击运行时实际执行的东西。

PyInstaller 需要一个**顶层脚本**来做依赖分析：如果把入口写成 `d4tracker/__main__.py`，
PyInstaller 会把 `d4tracker/` 目录当成脚本所在目录加进 sys.path，`import d4tracker`
反而找不到包。所以入口放在项目根。

这里还装了全局异常兜底：`--noconsole` 打包后崩溃只会弹一个没人看得清的系统框
（游戏一挡就没了），堆栈也就丢了。落到 `data/crash.log` 才能事后排查。
"""

from __future__ import annotations

import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _log_path() -> str:
    try:
        from d4tracker import paths
        return paths.log_path("crash.log")
    except Exception:                                    # noqa: BLE001
        return os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "crash.log")


def _report(exc_type, exc, tb) -> None:
    text = "".join(traceback.format_exception(exc_type, exc, tb))
    path = _log_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} "
                     f"(frozen={getattr(sys, 'frozen', False)}) =====\n{text}")
    except OSError:
        path = "(写日志失败)"

    # 没控制台时至少让用户看到一句话，并给出日志位置
    if sys.stdout is None:
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(
                None,
                f"{exc_type.__name__}: {exc}\n\n完整堆栈已写入:\n{path}",
                "D4 副本计数器 出错", 0x10)
        except Exception:                                # noqa: BLE001
            pass
    else:
        sys.stderr.write(text)
        sys.stderr.write(f"\n完整堆栈已写入: {path}\n")


def main() -> int:
    sys.excepthook = _report
    try:
        from d4tracker.monitor import main as run
    except Exception:                                    # noqa: BLE001
        _report(*sys.exc_info())
        return 1

    try:
        return run()
    except SystemExit:
        raise
    except BaseException:                                # noqa: BLE001
        _report(*sys.exc_info())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
