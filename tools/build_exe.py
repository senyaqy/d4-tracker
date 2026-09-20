"""把工具打包成 Windows exe（PyInstaller）。

几个非显然的选项，都是踩过才加的：

* ``--add-data templates`` —— 模板是**只读资源**，必须进包，否则 exe 拷到别的机器
  就找不到模板。数据库相反，走 `paths.data_dir()` 落到 exe 旁边（或 LOCALAPPDATA），
  绝不能进包 —— 单文件模式解压出来的临时目录退出即删。
* ``--collect-all rapidocr_onnxruntime`` —— 它的 .onnx 模型和配置是**数据文件**，
  PyInstaller 靠 import 分析看不到，不显式收集的话运行时会在找不到模型处崩掉。
* ``--noconsole`` —— 双击不弹黑框。代价是 ``sys.stdout`` 变 None，所以自检报告改走
  ``data/selftest.log``（见 monitor.run_selftest 的 TEE）。
* ``--exclude-module`` —— PySide6-Addons 装了 160MB+（QtWebEngine/Qt3D/QtCharts…），
  本工具只用 QtCore/QtGui/QtWidgets，不排除的话包会大出好几倍。

用法::

    .venv\\Scripts\\python.exe tools\\build_exe.py              # 单文件夹（启动快，推荐）
    .venv\\Scripts\\python.exe tools\\build_exe.py --onefile    # 单个 exe（便携，启动慢）
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENTRY = os.path.join(ROOT, "run_tracker.py")
ICON = os.path.join(ROOT, "assets", "d4tracker.ico")
DIST = os.path.join(ROOT, "dist")
BUILD = os.path.join(ROOT, "build")

EXCLUDES = [
    # 本工具用不到，但体积大或会拖慢分析
    "tkinter", "matplotlib", "pandas", "scipy", "IPython", "jupyter",
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebChannel",
    "PySide6.QtQml", "PySide6.QtQuick", "PySide6.QtQuick3D", "PySide6.Qt3DCore",
    "PySide6.QtCharts", "PySide6.QtDataVisualization", "PySide6.QtMultimedia",
    "PySide6.QtSql", "PySide6.QtTest", "PySide6.QtDesigner",
    "PySide6.QtPdf", "PySide6.QtPositioning", "PySide6.QtSerialPort",
    "PySide6.QtBluetooth", "PySide6.QtNfc", "PySide6.QtWebSockets", "PySide6.QtHelp",
    # 注意：不要排除 PySide6.QtNetwork / QtOpenGL / shiboken6.Shiboken。
    # 第一版排了，结果 GUI 起来就崩 —— QtGui/QtWidgets 会间接依赖它们，
    # 排掉之后 Qt 平台插件加载失败（selftest 不 import Qt，所以完全测不出来）。
    "onnxruntime.tools", "torch", "tensorflow",
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="打包 D4 副本计数器")
    ap.add_argument("--onefile", action="store_true", help="打成单个 exe（启动较慢）")
    ap.add_argument("--console", action="store_true",
                    help="保留控制台窗口，用于排查打包后的启动错误")
    ap.add_argument("--name", default="D4Tracker")
    ap.add_argument("--keep-build", action="store_true", help="保留 build/ 中间产物")
    args = ap.parse_args(argv)

    if not os.path.exists(ENTRY):
        print(f"找不到入口脚本 {ENTRY}")
        return 1

    # 先结束残留实例。它会占住 dist/<name>/data/counts.db（可写数据就在 exe 旁边），
    # PyInstaller 清理输出目录时就会报 WinError 32 —— 那个报错完全看不出是这个原因。
    if os.name == "nt":
        for image in {f"{args.name}.exe", "D4Tracker.exe", "D4Tracker-portable.exe"}:
            subprocess.call(["taskkill", "/IM", image, "/F"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm", "--clean",
        "--name", args.name,
        "--distpath", DIST,
        "--workpath", BUILD,
        "--specpath", BUILD,
        "--onefile" if args.onefile else "--onedir",
        "--console" if args.console else "--noconsole",
        "--paths", ROOT,
        # 必须用绝对路径：--add-data 的源路径是相对 **specpath** 解析的，不是 cwd。
        # 写成相对路径时，一旦 --specpath 指向 build/，就会去找 build/templates 而失败。
        "--add-data", f"{os.path.join(ROOT, 'templates')}{os.pathsep}templates",
        "--add-data", f"{os.path.join(ROOT, 'assets')}{os.pathsep}assets",
        # RapidOCR 的 onnx 模型是数据文件，import 分析看不到
        "--collect-all", "rapidocr_onnxruntime",
        # 隐藏导入：onnxruntime 的 provider 是按名字动态加载的
        "--hidden-import", "onnxruntime",
        "--hidden-import", "onnxruntime.capi._pybind_state",
    ]
    if os.path.exists(ICON):
        cmd += ["--icon", ICON]
    else:
        print(f"（没有图标文件 {ICON}，跳过；可先跑 tools/make_icon.py）")

    # 中文翻译只带需要的这一个：PyInstaller 的 PySide6 hook 不保证收集 translations，
    # 少了它 QMessageBox / QInputDialog 的标准按钮会退回 Yes / No / OK
    qm = os.path.join(sys.prefix, "Lib", "site-packages", "PySide6",
                      "translations", "qtbase_zh_CN.qm")
    if os.path.exists(qm):
        cmd += ["--add-data", f"{qm}{os.pathsep}translations"]
    else:
        print(f"（没找到 {qm}，弹框按钮会是英文）")

    for mod in EXCLUDES:
        cmd += ["--exclude-module", mod]
    cmd.append(ENTRY)

    print("执行: " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
    print()
    t0 = time.perf_counter()
    rc = subprocess.call(cmd, cwd=ROOT)
    dt = time.perf_counter() - t0
    print(f"\nPyInstaller 退出码 {rc}，耗时 {dt:.0f}s")

    if rc != 0:
        return rc

    target = os.path.join(DIST, f"{args.name}.exe" if args.onefile
                          else os.path.join(args.name, f"{args.name}.exe"))
    if os.path.exists(target):
        size = os.path.getsize(target) / 1024 / 1024
        print(f"产物: {target}  ({size:.1f} MB)")
        if not args.onefile:
            total = sum(os.path.getsize(os.path.join(dp, f))
                        for dp, _dn, fn in os.walk(os.path.join(DIST, args.name))
                        for f in fn)
            print(f"整目录: {total / 1024 / 1024:.1f} MB")

    if not args.keep_build and os.path.isdir(BUILD):
        shutil.rmtree(BUILD, ignore_errors=True)
        print(f"已清理中间产物 {BUILD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
