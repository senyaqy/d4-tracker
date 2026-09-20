"""固定位置的数值读取：地狱狂潮的畸变余烬 / 灾祸之心。

和"完成事件"是两类东西
----------------------
副本完成是一次性事件（出现→消失），靠上升沿计数。余烬和灾祸之心是**数值**：
随拾取增长、去祭坛时被消耗。所以这里不检测"完成"，而是周期性读数字，只在变化时落库。

位置（参考 2560x1440）
---------------------
小地图左侧那块面板：上面是倒计时 `地狱狂潮：48:30`，下面两个圆形图标
（左=火焰=畸变余烬，右=恶魔=灾祸之心），数字在图标正下方，字形约 12x23。

为什么绕过检测模型
------------------
RapidOCR 完整流程里 ``min_height`` 默认 30，而数字字形只有 **23 像素高**，会被高度
过滤器丢掉 —— 把 min_height 调到 6 也没用（实测 6/10/30 结果一样，检测模型根本没把
这两个小框提出来）。直接调 ``text_recognizer`` 只要几毫秒。

为什么单帧读数不能直接信
------------------------
字形是 D4 那种带装饰的字体，单字识别天然不稳：

* 固定阈值切分在亮场景下会把整块面板算成前景（实测 bbox 变成整个 ROI，读出 `57`）；
* 改局部 Otsu 后余烬稳了，但灾祸之心的 `7` 会有一半概率退化成 `1`。

所以采**多变体 + 时间窗口投票**：每帧用三种切分各读一次，把结果投进滑动窗口，
取窗口内众数，并要求众数明显领先第二名。实测三个样本六个槽位全部给出正确众数，
且优势很大（正确值 28~39 票，第二名最多 8 票）。
"""

from __future__ import annotations

import collections
from dataclasses import dataclass

import cv2
import numpy as np

from .detect import cut_roi

_ENGINE = None

# 全角数字 -> 半角
_FULLWIDTH = str.maketrans("０１２３４５６７８９", "0123456789")


def engine():
    global _ENGINE
    if _ENGINE is None:
        from rapidocr_onnxruntime import RapidOCR
        _ENGINE = RapidOCR()
    return _ENGINE


@dataclass(frozen=True)
class ValueSlot:
    key: str
    label: str
    roi: tuple[int, int, int, int]
    scale: int = 8
    max_digits: int = 4


# 数字位于图标正下方；ROI 比字形宽，给 1~4 位数留余量
HELLTIDE_SLOTS = (
    ValueSlot("cinders", "畸变余烬", (1895, 183, 80, 46)),
    ValueSlot("baneful", "灾祸之心", (2010, 183, 80, 46)),
)


def _recognize(gray: np.ndarray, scale: int, max_side: int = 256) -> str:
    """放大后送识别。

    放大倍数要卡上限：亮场景下 Otsu 的包围盒可能覆盖整个 ROI（80x46），再乘 8 就是
    640x368 —— 实测这种输入会让识别从十几毫秒飙到 300ms 以上（数值读取的"尖刺"
    就是它）。限制长边到 256 之后，两种情形都是十几毫秒。
    """
    h, w = gray.shape[:2]
    factor = min(float(scale), max(2.0, max_side / max(h, w)))
    big = cv2.resize(gray, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
    if big.ndim == 2:
        big = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
    try:
        res, _elapse = engine().text_recognizer([big])
    except Exception:                                    # noqa: BLE001
        return ""
    if not res:
        return ""
    text = str(res[0][0]).translate(_FULLWIDTH)
    return "".join(c for c in text if c.isdigit())


def read_variants(frame, slot: ValueSlot) -> list[str]:
    """用几种切分方式各读一次，返回候选字符串（可能为空）。

    三种切分不是冗余：实测它们在亮场景/暗场景下各有胜负，合起来才稳。
    """
    crop = cut_roi(frame, slot.roi)
    if crop.size == 0:
        return []

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    candidates: list[np.ndarray] = []

    # 1/2：Otsu 的亮侧与暗侧，各取紧致包围盒
    for invert in (False, True):
        binary = (gray > thresh).astype(np.uint8) * 255
        if invert:
            binary = 255 - binary
        ys, xs = np.nonzero(binary > 127)
        if len(xs):
            candidates.append(binary[ys.min():ys.max() + 1, xs.min():xs.max() + 1])

    # 3：包围盒内保留灰度，不做二值化 —— 二值化会把装饰字体的笔画切断
    ys, xs = np.nonzero(gray > thresh)
    if len(xs):
        candidates.append(gray[ys.min():ys.max() + 1, xs.min():xs.max() + 1])

    out = []
    for cand in candidates:
        h, w = cand.shape[:2]
        ratio = w / max(h, 1)
        # 极端长宽比一定不是数字（单字约 0.5，2~4 位数约 1~2）。而且识别模型是按
        # 高度归一化的：一个 80x5 的包围盒归一化后宽度会撑到 768，实测能跑到 300ms+，
        # 那些"尖刺"就是它。这条既省时间也提高了准确率。
        if ratio < 0.2 or ratio > 4.0:
            continue
        text = _recognize(cand, slot.scale)
        if text and len(text) <= slot.max_digits:
            out.append(text)
    return out


class ValueReader:
    """滑动窗口投票：单帧读数不可信，窗口众数才可信。"""

    def __init__(self, slots=HELLTIDE_SLOTS, window: int = 10,
                 min_votes: int = 4, lead: float = 2.0) -> None:
        self.slots = tuple(slots)
        self.window = window
        self.min_votes = min_votes
        self.lead = lead
        self.current: dict[str, int] = {}
        self.confidence: dict[str, float] = {}
        self._votes: dict[str, collections.deque] = {
            s.key: collections.deque(maxlen=window) for s in slots
        }

    def read(self, frame) -> dict[str, int]:
        """读一帧，返回**已确认**的数值（某项还没确认就不会出现在返回里）。"""
        for slot in self.slots:
            found = read_variants(frame, slot)
            self._votes[slot.key].append(collections.Counter(found))

            total: collections.Counter = collections.Counter()
            for vote in self._votes[slot.key]:
                total.update(vote)
            if not total:
                continue

            (best, best_n), *rest = total.most_common(2) or [(None, 0)]
            second_n = rest[0][1] if rest else 0
            if best_n < self.min_votes or best_n < self.lead * max(second_n, 1):
                continue

            self.current[slot.key] = int(best)
            self.confidence[slot.key] = best_n / max(1, sum(total.values()))

        return dict(self.current)

    def reset(self) -> None:
        for dq in self._votes.values():
            dq.clear()
        self.current.clear()
        self.confidence.clear()

    def labels(self) -> dict[str, str]:
        return {s.key: s.label for s in self.slots}


__all__ = ["ValueSlot", "ValueReader", "HELLTIDE_SLOTS", "read_variants"]
