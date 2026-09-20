"""生成 exe 图标（多尺寸 .ico）。

用程序生成而不是塞一张位图：exe 图标需要 16/32/48/256 多种尺寸，Windows 会按场景
挑用；只放一张大图缩放到 16x16 会糊成一团。

用法::

    .venv\\Scripts\\python.exe tools\\make_icon.py
"""

from __future__ import annotations

import os
import sys

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "assets")
OUT = os.path.join(OUT_DIR, "d4tracker.ico")

SIZES = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]

BG = (18, 18, 24, 255)
RING = (126, 22, 26, 255)
DIAMOND = (198, 44, 44, 255)
GLYPH = (238, 200, 96, 255)

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\arialbd.ttf",
    r"C:\Windows\Fonts\seguisb.ttf",
    r"C:\Windows\Fonts\msyhbd.ttc",
]


def load_font(size: int):
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def render(size: int = 512) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    pad = max(2, size // 32)
    radius = size // 5
    d.rounded_rectangle([pad, pad, size - pad, size - pad], radius=radius,
                        fill=BG, outline=RING, width=max(2, size // 42))

    cx = size // 2
    inset = size // 6
    d.polygon([(cx, inset), (size - inset, cx), (cx, size - inset), (inset, cx)],
              outline=DIAMOND, width=max(2, size // 48))

    font = load_font(int(size * 0.60))
    text = "4"
    bbox = d.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    d.text((cx - tw / 2 - bbox[0], cx - th / 2 - bbox[1]), text, font=font, fill=GLYPH)
    return img


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    base = render(512)
    base.save(OUT, format="ICO", sizes=SIZES)
    print(f"图标 -> {OUT}  ({os.path.getsize(OUT)} bytes, 尺寸 {[s[0] for s in SIZES]})")

    preview = os.path.join(OUT_DIR, "d4tracker-256.png")
    base.resize((256, 256), Image.LANCZOS).save(preview)
    print(f"预览 -> {preview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
