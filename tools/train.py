"""从样本里裁出横幅模板。

模板必须来自"文字确实显示着"的那一帧，所以每个条目都写明了来源，方便复查。

用法::

    .venv\\Scripts\\python.exe tools\\train.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from d4tracker import detect  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES = os.path.join(ROOT, "samples")

# 顶部居中横幅（梦魇 / 深坑）
BANNER = (1080, 45, 460, 85)

# (模板名, 显示标签, 样本目录, 帧号, ROI)
# ROI 全部由样本实测得到，不是猜的；位置写进 templates.json，按参考分辨率缩放。
# 文字/背景的分割阈值由建模板时对该 ROI 跑 Otsu 定下并固定，不需要手工指定极性。
# 巢穴首领：目标行会从 `击败<首领>的折磨回响` 变成 `打开<首领>的秘宝`。
# 首领名长度不同 -> 整串位置会移动，所以用"条带内滑动搜索"匹配不变的尾巴 `的秘宝`。
# 条带只盖第一条目标行（y 484~528）；下面常驻的加成说明里也有"秘宝"二字，
# 所以必须靠 y 分隔干净。
LAIR_STRIP = (1975, 484, 590, 44)
LAIR_QUERY = (2378, 486, 128, 40)

SPECS = [
    ("nightmare_complete", "梦魇地下城", "梦魇地下城-完成-20260920-210543", 0, BANNER),
    ("dungeon_complete", "深坑", "深坑-完成-20260920-205936", 14, BANNER),
    # 炼狱魔潮不能用右侧的 `已完成地下城` —— 那是**通用**的地下城完成标记，
    # 深坑样本里同样有，会把深坑算成魔潮（实测误报到 0.74）。
    # 改用波次计数器 `波次：10/10`（1900,50 附近），实测只有魔潮样本有。
    ("hordes_complete", "炼狱魔潮", "炼狱魔潮-完成-20260920-214532", 24, (1898, 45, 232, 56)),
    ("kurast_complete", "库拉斯特地下城", "库拉斯特地下城-完成-20260920-205019", 0, (2330, 603, 240, 62)),
    ("tree_turnin", "秘语之树", "秘语之树-交付-20260920-212808", 24, (1540, 685, 480, 130)),
    ("lair_complete", "巢穴首领", "巢穴首领-击杀-20260920-220211", 12, LAIR_STRIP, LAIR_QUERY),
]


def main() -> int:
    print(f"参考分辨率 {detect.REF_W}x{detect.REF_H}")
    print(f"输出目录 {detect.TEMPLATE_DIR}\n")
    items = detect.build_from_samples(SPECS, SAMPLES)
    print(f"\n共 {len(items)} 个模板 -> {detect.INDEX_PATH}")
    return 0 if items else 1


if __name__ == "__main__":
    raise SystemExit(main())
