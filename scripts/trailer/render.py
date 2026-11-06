"""Render the hands trailer frame by frame: every frame is trailer.frame(t) on the page, so timing is exact.

Usage:
  uv run --no-project --with playwright python -I scripts/trailer/render.py OUT.mp4
  uv run --no-project --with playwright python -I scripts/trailer/render.py --stills DIR 0.5 3.1 ...

Needs the trailer server (in web/): npx vite --config vite.trailer.config.ts, with web/trailer-public/assets linked
to the local game art (ln -s <repo>/artifacts/mobile-assets web/trailer-public/assets). TRAILER_PORT picks the port.
"""

import asyncio
import os
import subprocess
import sys
from pathlib import Path

from playwright.async_api import async_playwright

URL = f"http://127.0.0.1:{os.environ.get('TRAILER_PORT', '5197')}/trailer.html"
W, H, FPS = 1920, 1080, 60


async def open_page(p):
    browser = await p.chromium.launch(channel="chrome", headless=True, args=["--use-angle=metal"])
    page = await browser.new_page(viewport={"width": W, "height": H})
    page.on("console", lambda m: m.type == "error" and print("console:", m.text, file=sys.stderr))
    await page.goto(URL)
    await page.wait_for_function("window.trailer && window.trailer.ready", timeout=60000)
    return browser, page


async def stills(out, times):
    Path(out).mkdir(parents=True, exist_ok=True)
    async with async_playwright() as p:
        browser, page = await open_page(p)
        for t in times:
            await page.evaluate("t => trailer.frame(t)", t)
            await page.screenshot(path=f"{out}/t{t:05.2f}.png")
        await browser.close()


async def video(out):
    ff = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "image2pipe", "-framerate", str(FPS), "-c:v", "png", "-i", "-",
         "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out],
        stdin=subprocess.PIPE,
    )
    async with async_playwright() as p:
        browser, page = await open_page(p)
        duration = await page.evaluate("trailer.duration")
        frames = round(duration * FPS)
        for i in range(frames):
            await page.evaluate("t => trailer.frame(t)", i / FPS)
            ff.stdin.write(await page.screenshot(type="png"))
            if i % 120 == 0:
                print(f"{out}: frame {i}/{frames}", flush=True)
        await browser.close()
    ff.stdin.close()
    ff.wait()
    print(f"{out}: done, {frames} frames ({frames / FPS:.1f} s)")


if __name__ == "__main__":
    if sys.argv[1] == "--stills":
        asyncio.run(stills(sys.argv[2], [float(x) for x in sys.argv[3:]]))
    else:
        asyncio.run(video(sys.argv[1]))
