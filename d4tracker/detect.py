"""UI 文案识别：按亮度取文字骨架，再和模板比 IoU。

信号不止一处 —— 实测各类活动的"完成"表现：
--------------------------------------------------
* 梦魇地下城  -> **顶部居中横幅** ``梦魇地下城完成`` + 区域名
* 深坑        -> **顶部居中横幅** ``地下城完成``   + 区域名
* 炼狱魔潮    -> 右侧目标面板 ``已完成地下城``（波次打完后 4 秒以上）
* 库拉斯特    -> 右侧目标面板 ``区域通关``
* 秘语之树    -> 中央奖励面板 ``使用此宝匣获取你的奖励。``

所以模板各自带 ROI，而不是共用一个全局区域。位置由样本实测得到，用归一化坐标
存储以适配其它分辨率。

为什么用"文字掩膜 + IoU"而不是 cv2.matchTemplate 直接匹配
--------------------------------------------------------
这些面板背景是半透明的，背后是不断变化的火焰、粒子、怪物。直接对彩色/灰度图做
归一化互相关会把背景一起算进去，换个场景分数就掉。文字是接近纯白的亮色带深色
描边，所以先按亮度阈值取出骨架，再和模板骨架比 IoU（交并比）—— 背景怎么变都不
影响骨架。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import cv2
import numpy as np

from . import imageio, paths

# 模板是只读资源：打包后跟着 exe 走（sys._MEIPASS / exe 目录），源码运行时是项目根
TEMPLATE_DIR = os.path.join(paths.resource_root(), "templates")
INDEX_PATH = os.path.join(TEMPLATE_DIR, "templates.json")

# 参考分辨率（采集样本时是 2560x1440）
REF_W, REF_H = 2560, 1440

# 默认 ROI：顶部居中横幅
DEFAULT_ROI = (1080, 45, 460, 85)

TEXT_THRESHOLD = 190      # 暗底亮字：取亮于这个值的像素
DARK_THRESHOLD = 70       # 亮底暗字：取暗于这个值的像素

# 阈值来自 tools/validate.py 在全部样本上的实测（见 README「验证结果」）。
# 取 0.55：离反例天花板留足余量，同时兜住面板淡入淡出时的半透明帧
# （淡出时文字变淡、掩膜变弱，IoU 会掉到 0.6~0.8）。
MATCH_THRESHOLD = 0.55

# 连续命中多少帧才算一次事件。3fps 下 2 帧约 0.6 秒，
# 滤掉单帧噪点，又不至于漏掉只停留一秒多的横幅。
CONFIRM_FRAMES = 2


def otsu_split(bgr: np.ndarray) -> tuple[int, str, int]:
    """找文字/背景的分割阈值，并判定文字在哪一侧。返回 (阈值, 极性, 文字像素数)。

    固定阈值跨面板根本不可通用：顶部横幅是接近纯白的字（>190），而秘语之树奖励
    面板的字只有 120~171 —— 用 190 一个像素都取不到，用 70 又会把整片黑底当成文字
    （实测那会让"暗色像素"占到 ROI 的 86%，分离度从 0.80 塌到 0.14）。

    Otsu 按 ROI 自己的直方图找分割点（对这种"黑底 + 灰字"的双峰分布很准），
    再取**面积更小的那一侧**当文字 —— 文字永远比背景少。
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t = int(t)
    n_bright = int((gray > t).sum())
    n_dark = int((gray < t).sum())
    if n_bright <= n_dark:
        return t, "bright", n_bright
    return t, "dark", n_dark


def apply_split(bgr: np.ndarray, threshold: int, polarity: str) -> np.ndarray:
    """按存好的阈值/极性取文字掩膜。阈值来自建模板时的 Otsu，之后固定不变 ——
    自适应阈值会让同一块面板在不同帧上得到不同的掩膜，IoU 就没法比了。"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if polarity == "dark":
        return (gray < threshold).astype(np.uint8)
    return (gray > threshold).astype(np.uint8)


def text_mask(bgr: np.ndarray, polarity: str = "bright",
              threshold: int | None = None) -> np.ndarray:
    """兼容入口：不给阈值时退回固定阈值。"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if polarity == "dark":
        return (gray < (DARK_THRESHOLD if threshold is None else threshold)).astype(np.uint8)
    return (gray > (TEXT_THRESHOLD if threshold is None else threshold)).astype(np.uint8)


def scale_rect(rect, frame_w: int, frame_h: int, ref_w: int = REF_W, ref_h: int = REF_H):
    x, y, w, h = rect
    sx, sy = frame_w / ref_w, frame_h / ref_h
    return (int(round(x * sx)), int(round(y * sy)),
            max(1, int(round(w * sx))), max(1, int(round(h * sy))))


@dataclass
class Template:
    name: str
    label: str
    roi: tuple[int, int, int, int]      # 参考分辨率下的 ROI（搜索型则是查询片段的位置）
    mask: np.ndarray                    # 文字掩膜
    threshold: int = TEXT_THRESHOLD     # 建模板时 Otsu 定下的分割阈值
    polarity: str = "bright"            # 'bright' = 取亮于阈值的一侧
    search: bool = False                # True = 在 strip 内滑动搜索，而非固定 ROI 比对
    strip: tuple | None = None          # 搜索范围（参考分辨率）
    source: str = ""


def cut_roi(frame: np.ndarray, rect) -> np.ndarray:
    """按参考分辨率缩放后切一块 ROI。

    返回**连续数组**：抓帧给过来的可能是零拷贝步长视图，而 cv2 对非连续输入
    要么内部再拷一次、要么走慢路径。这里只拷贝小 ROI，值得。
    """
    h, w = frame.shape[:2]
    x, y, rw, rh = scale_rect(rect, w, h)
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    rw = max(1, min(rw, w - x))
    rh = max(1, min(rh, h - y))
    return np.ascontiguousarray(frame[y:y + rh, x:x + rw])


class Detector:
    """加载模板并对整帧分类。"""

    def __init__(self, template_dir: str = TEMPLATE_DIR) -> None:
        self.template_dir = template_dir
        self.ref_w, self.ref_h = REF_W, REF_H
        self.templates: list[Template] = []
        self._load()

    def _load(self) -> None:
        if not os.path.exists(INDEX_PATH):
            return
        with open(INDEX_PATH, "r", encoding="utf-8") as fh:
            index = json.load(fh)

        self.ref_w = index.get("ref_w", REF_W)
        self.ref_h = index.get("ref_h", REF_H)

        for item in index.get("items", []):
            path = os.path.join(self.template_dir, item["file"])
            mask = imageio.imread(path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            self.templates.append(Template(
                name=item["name"],
                label=item.get("label", item["name"]),
                roi=tuple(item.get("roi", DEFAULT_ROI)),
                mask=(mask > 127).astype(np.uint8),
                threshold=int(item.get("threshold", TEXT_THRESHOLD)),
                polarity=item.get("polarity", "bright"),
                search=bool(item.get("search", False)),
                strip=tuple(item["strip"]) if item.get("strip") else None,
                source=item.get("source", ""),
            ))

    @property
    def ready(self) -> bool:
        return len(self.templates) > 0

    def names(self) -> list[str]:
        return [t.name for t in self.templates]

    def rect_of(self, name: str):
        for t in self.templates:
            if t.name == name:
                return t.strip if (t.search and t.strip) else t.roi
        return DEFAULT_ROI

    def preview(self, frame: np.ndarray, name: str | None = None) -> np.ndarray | None:
        """取某个模板的 ROI 原图，给界面做"检测器看到了什么"的预览。"""
        rect = self.rect_of(name) if name else DEFAULT_ROI
        roi = cut_roi(frame, rect)
        return roi if roi.size else None

    def classify(self, frame: np.ndarray) -> list[tuple[str, str, float]]:
        """返回 [(name, label, iou)]，按分数降序。"""
        if not self.templates:
            return []

        out = []
        for t in self.templates:
            if t.search and t.strip:
                out.append((t.name, t.label, self._search(frame, t)))
            else:
                out.append((t.name, t.label, self._compare(frame, t)))

        out.sort(key=lambda r: r[2], reverse=True)
        return out

    def _compare(self, frame: np.ndarray, t: Template) -> float:
        """固定 ROI：整块比对 IoU。"""
        roi = cut_roi(frame, t.roi)
        if roi.size == 0:
            return 0.0
        return self._iou(apply_split(roi, t.threshold, t.polarity), t.mask)

    def _search(self, frame: np.ndarray, t: Template) -> float:
        """条带内滑动搜索：模板文字本身的位置会变，不能固定比对。

        巢穴首领的完成提示是 `打开<首领名>的秘宝` —— 首领名长度不同，整串的起止
        位置就不同，固定 ROI 一定对不齐。所以只在条带里找那个不变的尾巴 `的秘宝`。

        评分必须在**最佳位置上算真正的 IoU**。第一版只算召回（模板像素被覆盖的
        比例），结果在密集文字处随便对上一部分就能拿 0.58~0.70，几乎所有巢穴首领
        样本都被误报。把并集也计进来才分得开。
        """
        strip = cut_roi(frame, t.strip)
        if strip.size == 0:
            return 0.0

        mask = apply_split(strip, t.threshold, t.polarity).astype(np.float32)
        query = t.mask.astype(np.float32)
        if mask.shape[0] < query.shape[0] or mask.shape[1] < query.shape[1]:
            return 0.0

        qn = float(query.sum())
        if qn <= 0:
            return 0.0

        inter = cv2.matchTemplate(mask, query, cv2.TM_CCORR)
        local = cv2.matchTemplate(mask, np.ones_like(query), cv2.TM_CCORR)
        union = qn + local - inter
        iou = inter / np.maximum(union, 1e-6)
        return float(min(1.0, iou.max()))

    @staticmethod
    def _iou(mask: np.ndarray, tmpl: np.ndarray) -> float:
        if tmpl.shape != mask.shape:
            tmpl = cv2.resize(tmpl, (mask.shape[1], mask.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
            tmpl = (tmpl > 0).astype(np.uint8)

        inter = int(np.logical_and(mask, tmpl).sum())
        union = int(np.logical_or(mask, tmpl).sum())
        iou = inter / union if union else 0.0

        fp, tp = int(mask.sum()), int(tmpl.sum())
        if tp and (fp < tp * 0.4 or fp > tp * 3.0):
            iou *= 0.3
        return iou


def build_from_samples(specs, samples_root: str, out_dir: str = TEMPLATE_DIR) -> list[dict]:
    """从样本里裁出模板掩膜并写盘。

    specs: [(name, label, 样本目录名, 帧号, ROI), ...]
    """
    os.makedirs(out_dir, exist_ok=True)
    items = []

    for spec in specs:
        name, label, folder, frame = spec[0], spec[1], spec[2], spec[3]
        roi = tuple(spec[4]) if len(spec) > 4 and spec[4] else DEFAULT_ROI
        # 给第 6 项 = 查询片段的位置 -> 变成"条带内滑动搜索"型模板
        query = tuple(spec[5]) if len(spec) > 5 and spec[5] else None
        source_rect = query or roi

        path = os.path.join(samples_root, folder, f"f{frame:03d}.jpg")
        img = imageio.imread(path)
        if img is None:
            print(f"  跳过 {name}: 读不到 {path}")
            continue

        h, w = img.shape[:2]
        if (w, h) != (REF_W, REF_H):
            print(f"  警告 {name}: 样本是 {w}x{h}，与参考 {REF_W}x{REF_H} 不同，模板可能对不齐")

        crop = cut_roi(img, source_rect)
        threshold, polarity, pixels = otsu_split(crop)
        mask = apply_split(crop, threshold, polarity) * 255

        density = pixels / max(1, mask.shape[0] * mask.shape[1])
        if pixels < 200 or density > 0.6:
            print(f"  !! {name}: 文字像素 {pixels}（占 {density:.0%}），"
                  f"ROI 多半写错了 —— 文字应该明显少于背景")

        file = f"{name}.png"
        if not imageio.imwrite(os.path.join(out_dir, file), mask):
            print(f"  跳过 {name}: 写模板失败")
            continue

        item = {
            "name": name, "label": label, "file": file,
            "roi": list(source_rect), "pixels": pixels,
            "threshold": threshold, "polarity": polarity,
            "density": round(density, 3),
            "source": f"{folder}#{frame}",
        }
        if query:
            item["search"] = True
            item["strip"] = list(roi)
        items.append(item)

        mode = f"搜索 strip={roi}" if query else "固定 ROI"
        print(f"  模板 {name:22s} {mode:28s} 查询框={str(source_rect):22s} "
              f"Otsu={threshold:3d} {polarity:6s} 像素 {pixels:5d}  来自 {folder}#{frame}")

    index = {
        "ref_w": REF_W, "ref_h": REF_H,
        "text_threshold": TEXT_THRESHOLD,
        "match_threshold": MATCH_THRESHOLD,
        "confirm_frames": CONFIRM_FRAMES,
        "items": items,
    }
    with open(INDEX_PATH, "w", encoding="utf-8") as fh:
        json.dump(index, fh, ensure_ascii=False, indent=2)
    return items


# 兼容旧名字
BannerDetector = Detector

__all__ = ["Detector", "BannerDetector", "Template", "build_from_samples",
           "text_mask", "otsu_split", "apply_split",
           "DEFAULT_ROI", "BANNER_ROI", "TEMPLATE_DIR", "INDEX_PATH", "scale_rect",
           "cut_roi", "MATCH_THRESHOLD", "CONFIRM_FRAMES"]

# 旧代码引用的名字
BANNER_ROI = DEFAULT_ROI
