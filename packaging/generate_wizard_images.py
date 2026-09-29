"""
Generate high-DPI Wizard images for Inno Setup 7.
- wizard_small: transparent RGBA PNGs (58x58, 124x124, 159x159)
- wizard_side: high-DPI 24-bit BMP banners (202x386, 430x824, 534x1022)
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter


def generate():
    pkg_dir = Path(__file__).resolve().parent
    repo_root = pkg_dir.parent
    logo_path = repo_root / "src" / "behavython" / "gui" / "assets" / "images" / "logo.png"
    ico_path = repo_root / "src" / "behavython" / "gui" / "assets" / "images" / "VY.ico"

    # 1. Wizard Small (Transparent High-DPI PNGs)
    small_sizes = {
        "100": (58, 58),
        "200": (124, 124),
        "250": (159, 159),
    }
    ico_img = Image.open(ico_path).convert("RGBA")
    ico_bbox = ico_img.getbbox()
    if ico_bbox:
        ico_img = ico_img.crop(ico_bbox)

    for scale, (w, h) in small_sizes.items():
        canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        pad = int(min(w, h) * 0.08)
        target_w = w - 2 * pad
        target_h = h - 2 * pad

        resized = ico_img.copy()
        resized.thumbnail((target_w, target_h), Image.Resampling.LANCZOS)

        offset_x = (w - resized.width) // 2
        offset_y = (h - resized.height) // 2
        canvas.paste(resized, (offset_x, offset_y), resized)

        out_png = pkg_dir / f"wizard_small_{scale}.png"
        canvas.save(out_png, format="PNG")
        print(f"Saved {out_png.name} ({w}x{h}) [Transparent PNG]")

    # 2. Wizard Side Banner (High-DPI 24-bit BMP)
    side_sizes = {
        "100": (202, 386),
        "200": (430, 824),
        "250": (534, 1022),
    }
    logo_img = Image.open(logo_path).convert("RGBA")
    logo_bbox = logo_img.getbbox()
    if logo_bbox:
        logo_img = logo_img.crop(logo_bbox)

    # Tilted orientation (90 deg): 'B' starts at top-left, slanting naturally down to 'N'
    tilted_logo = logo_img.rotate(80, expand=True, resample=Image.Resampling.BICUBIC)

    for scale, (w, h) in side_sizes.items():
        bg = Image.new("RGBA", (w, h))
        draw = ImageDraw.Draw(bg)
        top_color = (24, 30, 48)
        bot_color = (13, 18, 30)

        # Deep slate vertical gradient
        for y in range(h):
            ratio = y / h
            r = int(top_color[0] + (bot_color[0] - top_color[0]) * ratio)
            g = int(top_color[1] + (bot_color[1] - top_color[1]) * ratio)
            b = int(top_color[2] + (bot_color[2] - top_color[2]) * ratio)
            draw.line([(0, y), (w, y)], fill=(r, g, b, 255))

        # Fit logo proportionally inside max bounds so NO edge or letter is ever truncated
        max_h = int(h * 0.78)
        max_w = int(w * 0.72)
        scaled_logo = tilted_logo.copy()
        scaled_logo.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)

        pos_x = (w - scaled_logo.width) // 2
        pos_y = (h - scaled_logo.height) // 2

        # Subtle Gaussian drop shadow
        shadow = Image.new("RGBA", bg.size, (0, 0, 0, 0))
        shadow_offset = max(2, int(w * 0.015))
        shadow.paste(
            (0, 0, 0, 140),
            (pos_x + shadow_offset, pos_y + shadow_offset + 1),
            scaled_logo,
        )
        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=max(2, shadow_offset)))

        bg = Image.alpha_composite(bg, shadow)
        bg.paste(scaled_logo, (pos_x, pos_y), scaled_logo)

        out_bmp = pkg_dir / f"wizard_side_{scale}.bmp"
        bg.convert("RGB").save(out_bmp, format="BMP")
        print(f"Saved {out_bmp.name} ({w}x{h}) [BMP]")


if __name__ == "__main__":
    generate()
