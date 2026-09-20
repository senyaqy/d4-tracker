"""Windows 上对非 ASCII 路径安全的图像读写。

OpenCV 的 ``cv2.imread`` / ``cv2.imwrite`` 内部拿 ANSI 路径去 fopen：

* **读**：失败，返回 ``None``（配 ``IMREAD_COLOR`` 时还可能抛异常）；
* **写**：**静默失败** —— 返回 ``False``，不抛异常，文件根本不出现。

本项目的目录名全是中文（样本标签就是中文），所以这两个函数必须绕开 OpenCV 自带
的文件层，改成"Python 读/写字节 + cv2 解码/编码"。

踩过两次之后集中放这里：落盘（sampler）和分析（tools/contact）都走它。
"""

from __future__ import annotations

import os
import typing

import cv2
import numpy as np


def imread(path: str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """等价于 ``cv2.imread``，但中文路径可用。"""
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    try:
        return cv2.imdecode(data, flags)
    except cv2.error:
        return None


def imwrite(path: str, img: np.ndarray, params: typing.Sequence[int] | None = None) -> bool:
    """等价于 ``cv2.imwrite``，但中文路径可用；失败会如实返回 False。"""
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img, list(params) if params else [])
    if not ok:
        return False
    try:
        with open(path, "wb") as fh:
            fh.write(buf.tobytes())
    except OSError:
        return False
    return True


__all__ = ["imread", "imwrite"]
