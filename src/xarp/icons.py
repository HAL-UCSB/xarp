import io
from functools import lru_cache

import requests
from PIL import Image, ImageDraw, ImageFont, ImageOps

from xarp.entities import ImageAsset

MATERIAL_SYMBOLS_CODEPOINTS_URL = (
    "https://raw.githubusercontent.com/google/material-design-icons/master/"
    "variablefont/MaterialSymbolsOutlined[FILL,GRAD,opsz,wght].codepoints"
)
MATERIAL_SYMBOLS_FONT_URL = (
    "https://github.com/google/material-design-icons/raw/master/"
    "variablefont/MaterialSymbolsOutlined[FILL,GRAD,opsz,wght].ttf"
)


@lru_cache(maxsize=1)
def _material_symbol_codepoints() -> dict[str, str]:
    response = requests.get(MATERIAL_SYMBOLS_CODEPOINTS_URL, timeout=10)
    response.raise_for_status()
    return dict(
        line.split(maxsplit=1)
        for line in response.text.strip().splitlines()
        if line.strip()
    )


@lru_cache(maxsize=1)
def _material_symbol_font_bytes() -> bytes:
    response = requests.get(MATERIAL_SYMBOLS_FONT_URL, timeout=10)
    response.raise_for_status()
    return response.content


def _normalize_rgba(color: tuple[float, ...] | str) -> tuple[int, int, int, int] | str:
    if isinstance(color, str):
        return color

    r, g, b = color[:3]
    a = color[3] if len(color) == 4 else 1.0
    return (
        int(r * 255),
        int(g * 255),
        int(b * 255),
        int(a * 255),
    )


def material_symbol_image(
    name: str,
    size: int = 64,
    color: tuple[float, ...] | str = (0, 0, 0, 0.54),
    mirrored: bool = True,
) -> Image.Image:
    codepoints = _material_symbol_codepoints()
    if name not in codepoints:
        raise ValueError(f"Icon '{name}' not found. Browse at https://fonts.google.com/icons")

    char = chr(int(codepoints[name], 16))
    font = ImageFont.truetype(io.BytesIO(_material_symbol_font_bytes()), size=size)

    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.text(
        (size // 2, size // 2),
        char,
        font=font,
        fill=_normalize_rgba(color),
        anchor="mm",
    )

    if mirrored:
        img = ImageOps.mirror(img)

    return img


def material_symbol_asset(
    name: str,
    size: int = 64,
    color: tuple[float, ...] | str = (0, 0, 0, 0.54),
    mirrored: bool = True,
    asset_key: str | None = None,
) -> ImageAsset:
    return ImageAsset.from_obj(
        material_symbol_image(name, size=size, color=color, mirrored=mirrored),
        asset_key=asset_key,
    )
