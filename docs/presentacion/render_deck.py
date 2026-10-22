"""Renderiza el deck (deck.json + slides/*.html) a PNG con Edge headless y lo junta en PDF y PPTX.

    uv run --no-project --with python-pptx --with pillow python docs/presentacion/render_deck.py docs/presentacion/fuente <out_dir>
"""
import json
import os
import re
import subprocess
import sys

from PIL import Image

EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
ICONS = {"Chart": "📊", "Code": "💻", "Book": "📘", "Warning": "⚠️", "Lightbulb": "💡"}

HEAD = """<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Oswald:wght@400..700&family=Public+Sans:wght@400;600;700&display=swap" rel="stylesheet">
<style>
html,body{margin:0;padding:0;width:1920px;height:1080px;overflow:hidden}
section{position:relative;width:1920px;height:1080px;box-sizing:border-box;overflow:hidden}
p,h1,h2,h3,ul{margin:0}
ul{padding-left:1.2em}
aside{display:none}
table{border-collapse:collapse;width:100%}
th,td{padding:12px 16px;border-bottom:1px solid #D9D4C7;text-align:left;vertical-align:top}
th{font-weight:700}
.arrow{clip-path:polygon(0 30%,65% 30%,65% 0,100% 50%,65% 100%,65% 70%,0 70%)}
</style></head><body>
"""


def convert(html: str) -> str:
    def icon(m):
        style = m.group(2)
        size = re.search(r"height:(\d+)px", style)
        return '<span style="%s;font-size:%spx;line-height:1">%s</span>' % (
            style, int(size.group(1)) * 0.8 if size else 48, ICONS.get(m.group(1), "•"))
    html = re.sub(r'<x-icon name="(\w+)" style="([^"]*)"></x-icon>', icon, html)
    html = re.sub(r'<x-shape kind="ellipse" style="([^"]*)"></x-shape>',
                  r'<div style="\1;border-radius:50%;flex:none"></div>', html)
    html = re.sub(r'<x-shape kind="arrow-right" style="([^"]*)"></x-shape>',
                  r'<div class="arrow" style="\1;flex:none"></div>', html)
    return HEAD + html + "</body></html>"


def main():
    project, out = sys.argv[1], sys.argv[2]
    os.makedirs(os.path.join(out, "html"), exist_ok=True)
    os.makedirs(os.path.join(out, "png"), exist_ok=True)
    deck = json.load(open(os.path.join(project, "deck.json"), encoding="utf-8"))
    pngs = []
    for n, sid in enumerate(deck["order"], 1):
        src = open(os.path.join(project, "slides", sid + ".html"), encoding="utf-8").read()
        page = os.path.abspath(os.path.join(out, "html", "%02d_%s.html" % (n, sid)))
        open(page, "w", encoding="utf-8").write(convert(src))
        png = os.path.abspath(os.path.join(out, "png", "%02d_%s.png" % (n, sid)))
        subprocess.run([EDGE, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                        "--force-device-scale-factor=1", "--window-size=1920,1080",
                        "--virtual-time-budget=6000", "--screenshot=" + png,
                        "file:///" + page.replace("\\", "/")], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
        pngs.append(png)
        print("ok", n, sid)

    title = "La_MasIA_presentacion_interna"
    images = [Image.open(p).convert("RGB") for p in pngs]
    images[0].save(os.path.join(out, title + ".pdf"), save_all=True, append_images=images[1:],
                   resolution=150)

    from pptx import Presentation
    from pptx.util import Emu
    prs = Presentation()
    prs.slide_width, prs.slide_height = Emu(12192000), Emu(6858000)  # 16:9
    for p in pngs:
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        slide.shapes.add_picture(p, 0, 0, width=prs.slide_width, height=prs.slide_height)
    prs.save(os.path.join(out, title + ".pptx"))
    print("pdf y pptx en", out)


if __name__ == "__main__":
    main()
