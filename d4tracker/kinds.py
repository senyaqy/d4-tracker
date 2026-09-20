"""目标活动登记表。

界面要能显示"这项今天 0 次"，所以完整清单必须独立于"当前有没有模板"存在 ——
没模板的项显示为待标定，而不是从列表里消失。
"""

from __future__ import annotations

# (kind, 显示名, 是否有模板/能否自动识别, 备注)
TARGETS = [
    ("nightmare_complete", "梦魇地下城", True, "顶部居中横幅 `梦魇地下城完成`"),
    ("dungeon_complete", "深坑", True, "顶部居中横幅 `地下城完成`"),
    ("hordes_complete", "炼狱魔潮", True, "右侧目标面板 `已完成地下城`"),
    ("kurast_complete", "库拉斯特地下城", True, "右侧目标面板 `区域通关`"),
    ("tree_turnin", "秘语之树", True, "中央奖励面板 `使用此宝匣获取你的奖励。`"),
    ("lair_complete", "巢穴首领", True,
     "目标行从 `击败<首领>的折磨回响` 变成 `打开<首领>的秘宝`（条带搜索 `的秘宝`）"),
]

# 自定义条目的 kind：只手动记录，不参与自动识别
CUSTOM_KIND = "custom"

BY_KIND = {k: (label, ready, note) for k, label, ready, note in TARGETS}


def label_of(kind: str) -> str:
    return BY_KIND.get(kind, (kind, False, ""))[0]


# 样本目录名前缀 -> 期望命中的 kind。验证/回放都靠它判断"正例但类型认错"，
# 这类错误比单纯的漏报更隐蔽 —— 实测「已完成地下城」是通用标记，深坑也有，
# 险些把深坑全算成炼狱魔潮。
SAMPLE_EXPECT = {
    "梦魇地下城-完成": "nightmare_complete",
    "深坑-完成": "dungeon_complete",
    "炼狱魔潮-完成": "hordes_complete",
    "库拉斯特地下城-完成": "kurast_complete",
    "秘语之树-交付": "tree_turnin",
    "巢穴首领-击杀": "lair_complete",
}


def expected_kind(folder: str) -> str | None:
    """该样本目录应当命中哪个模板；没列出的一律视为"不该命中任何模板"。"""
    for prefix, kind in SAMPLE_EXPECT.items():
        if folder.startswith(prefix):
            return kind
    return None


def ready_kinds() -> list[str]:
    return [k for k, _, ready, _ in TARGETS if ready]


def all_kinds() -> list[str]:
    return [k for k, _, _, _ in TARGETS]


__all__ = ["TARGETS", "BY_KIND", "label_of", "ready_kinds", "all_kinds",
           "SAMPLE_EXPECT", "expected_kind", "CUSTOM_KIND"]
