"""Isometric PNG previews of map layouts (original simple art; no publisher graphics)."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .variants import (SHALLOW, OCEAN, FIELD, MOUNTAIN, FOREST, VILLAGE, RUIN, LIGHTHOUSE,
                       ANIMAL, FISH, METAL, FRUIT, STARFISH)

TW, TH = 44, 22
COLORS = {SHALLOW: (110, 190, 225), OCEAN: (38, 92, 160), FIELD: (170, 205, 110), MOUNTAIN: (150, 140, 130),
          FOREST: (95, 160, 85)}
PLAYERS = [(40, 120, 200), (210, 70, 60)]
RES = {ANIMAL: (120, 80, 40), FISH: (235, 250, 255), METAL: (60, 60, 70), FRUIT: (230, 60, 90), STARFISH: (255, 210, 60)}


def _font(size):
    for name in ('DejaVuSans.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 'Arial.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def render(layout, size, title='', subtitle=''):
    pad_top = 58
    w = size * TW + 20
    h = size * TH + 60 + pad_top
    img = Image.new('RGB', (w, h), (18, 22, 30))
    draw = ImageDraw.Draw(img)
    ox, oy = w // 2, pad_top + 10
    def center(i):
        x, y = i % size, i // size
        return ox + (x - y) * TW // 2, oy + (x + y) * TH // 2 + TH // 2
    caps = list(map(int, layout.capitals))
    order = sorted(range(size * size), key=lambda i: (i % size + i // size, i % size))
    for i in order:
        cx, cy = center(i)
        t = int(layout.terrain[i])
        base = COLORS.get(t, (90, 90, 90))
        poly = [(cx, cy - TH // 2), (cx + TW // 2, cy), (cx, cy + TH // 2), (cx - TW // 2, cy)]
        draw.polygon(poly, fill=base, outline=tuple(max(0, c - 35) for c in base))
        if t >= FIELD:  # little side face for land
            draw.polygon([(cx - TW // 2, cy), (cx, cy + TH // 2), (cx, cy + TH // 2 + 5), (cx - TW // 2, cy + 5)],
                         fill=tuple(max(0, c - 55) for c in base))
            draw.polygon([(cx + TW // 2, cy), (cx, cy + TH // 2), (cx, cy + TH // 2 + 5), (cx + TW // 2, cy + 5)],
                         fill=tuple(max(0, c - 75) for c in base))
        if t == MOUNTAIN:
            draw.polygon([(cx - 13, cy + 4), (cx, cy - 18), (cx + 13, cy + 4)], fill=(120, 112, 105), outline=(80, 75, 70))
            draw.polygon([(cx - 4, cy - 10), (cx, cy - 18), (cx + 4, cy - 10)], fill=(245, 245, 250))
        if t == FOREST:
            for dx, dy in ((-8, 0), (6, -2), (-1, 4)):
                draw.polygon([(cx + dx - 5, cy + dy + 2), (cx + dx, cy + dy - 11), (cx + dx + 5, cy + dy + 2)],
                             fill=(40, 105, 55), outline=(25, 70, 35))
        s = int(layout.structure[i])
        r = int(layout.resource[i])
        if r in RES:
            draw.ellipse([cx + 8, cy - 3, cx + 15, cy + 4], fill=RES[r], outline=(20, 20, 20))
        if s == VILLAGE:
            owner = caps.index(i) if i in caps else None
            if owner is not None:
                col = PLAYERS[owner]
                draw.rectangle([cx - 9, cy - 16, cx + 9, cy + 2], fill=col, outline=(255, 255, 255), width=2)
                for k in (-9, -3, 3):
                    draw.rectangle([cx + k, cy - 21, cx + k + 5, cy - 16], fill=col, outline=(255, 255, 255))
            else:
                draw.rectangle([cx - 6, cy - 9, cx + 6, cy + 1], fill=(235, 225, 200), outline=(90, 70, 50))
                draw.polygon([(cx - 8, cy - 9), (cx, cy - 16), (cx + 8, cy - 9)], fill=(170, 80, 60))
        elif s == RUIN:
            draw.rectangle([cx - 6, cy - 8, cx - 2, cy + 1], fill=(170, 170, 160))
            draw.rectangle([cx + 2, cy - 5, cx + 6, cy + 1], fill=(170, 170, 160))
        elif s == LIGHTHOUSE:
            draw.rectangle([cx - 3, cy - 16, cx + 3, cy + 1], fill=(245, 245, 245), outline=(200, 50, 50))
            draw.rectangle([cx - 3, cy - 20, cx + 3, cy - 16], fill=(255, 220, 80))
    draw.text((12, 8), title, fill=(255, 255, 255), font=_font(20))
    draw.text((12, 34), subtitle, fill=(185, 195, 210), font=_font(12))
    return img


def legend():
    img = Image.new('RGB', (900, 40), (18, 22, 30))
    d = ImageDraw.Draw(img)
    f = _font(13)
    items = [('field', COLORS[FIELD]), ('forest', COLORS[FOREST]), ('mountain', COLORS[MOUNTAIN]),
             ('shallow water', COLORS[SHALLOW]), ('ocean', COLORS[OCEAN]), ('blue capital', PLAYERS[0]),
             ('red capital', PLAYERS[1]), ('village', (235, 225, 200)), ('fruit', RES[FRUIT]), ('animal', RES[ANIMAL]),
             ('fish', RES[FISH]), ('metal', RES[METAL])]
    x = 10
    for name, col in items:
        d.rectangle([x, 13, x + 14, 27], fill=col, outline=(200, 200, 200))
        d.text((x + 18, 12), name, fill=(220, 225, 235), font=f)
        x += 26 + int(d.textlength(name, font=f))
    return img


def sheet(images, columns=3, header=None):
    w, h = images[0].size
    rows = (len(images) + columns - 1) // columns
    top = header.size[1] if header else 0
    out = Image.new('RGB', (w * columns, h * rows + top), (12, 15, 20))
    if header:
        out.paste(header, (0, 0))
    for k, im in enumerate(images):
        out.paste(im, ((k % columns) * w, top + (k // columns) * h))
    return out
