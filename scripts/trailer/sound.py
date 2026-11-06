"""Score the hands trailer: read the cue sheet from the page (trailer.soundEvents()), synthesize a stereo track
to it, and mux it into a rendered video.

One music track, synthesized here (no samples): a pad chord per game (Am, F, C, G) under a soft beat and a bassline
on the chord's root, a riser building into each mate, one chord hit with a crash on the win, where the beat drops
out until the flipped board lands, and the pad carrying on alone under the closing leaderboard.

Usage (trailer server running, as for render.py):
  uv run --no-project --with playwright --with numpy --with scipy python -I scripts/trailer/sound.py IN.mp4 OUT.mp4
"""

import asyncio
import os
import subprocess
import sys
import tempfile

import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve, lfilter

URL = f"http://127.0.0.1:{os.environ.get('TRAILER_PORT', '5197')}/trailer.html"
SR = 48000
rng = np.random.default_rng(7)


async def cues():
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(channel="chrome", headless=True)
        page = await browser.new_page(viewport={"width": 1920, "height": 1080})
        await page.goto(URL, wait_until="domcontentloaded", timeout=90000)
        await page.wait_for_function("window.trailer && window.trailer.ready", timeout=60000)
        ev = await page.evaluate("trailer.soundEvents()")
        await browser.close()
    return ev


# ---------- building blocks ----------
def tt(dur):
    return np.arange(int(dur * SR)) / SR


def hz(note):  # MIDI note -> Hz
    return 440.0 * 2 ** ((note - 69) / 12)


def env(t, attack, decay):
    return np.minimum(1, t / max(attack, 1e-4)) * np.exp(-t / decay)


def lowpass(x, fc):
    """One-pole low-pass; fc may be a per-sample array (Hz)."""
    if np.isscalar(fc):
        a = 1 - np.exp(-2 * np.pi * fc / SR)
        return lfilter([a], [1, a - 1], x)
    a = 1 - np.exp(-2 * np.pi * np.broadcast_to(fc, x.shape) / SR)
    y = np.empty_like(x)
    acc = 0.0
    for i in range(len(x)):
        acc += a[i] * (x[i] - acc)
        y[i] = acc
    return y


def highpass(x, fc):
    return x - lowpass(x, fc)


def saw(f, t):
    return 2 * ((f * t) % 1) - 1




def noise(dur):
    return rng.uniform(-1, 1, int(dur * SR))


class Mix:
    def __init__(self, duration):
        self.n = int(duration * SR)
        self.dry = np.zeros((2, self.n))
        self.send = np.zeros((2, self.n))  # into the reverb

    def add(self, t, x, gain=1.0, pan=0.0, verb=0.0):
        i = int(t * SR)
        if i >= self.n or len(x) == 0:
            return
        if i < 0:
            x, i = x[-i:], 0
        x = x[: self.n - i] * gain
        left, right = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
        for ch, g in ((0, left), (1, right)):
            self.dry[ch, i : i + len(x)] += x * g * (1 - verb * 0.5)
            self.send[ch, i : i + len(x)] += x * g * verb

    def master(self, fade_in, fade_out_from):
        ir_t = tt(2.6)
        ir = [noise(2.6) * np.exp(-ir_t / 0.55) for _ in range(2)]
        ir = [lowpass(r, 5000) for r in ir]
        # Only the mids and highs go into the reverb, so the low end stays tight.
        send = [highpass(highpass(self.send[c], 250), 250) for c in range(2)]
        wet = np.stack([fftconvolve(send[c], ir[c])[: self.n] for c in range(2)])
        out = self.dry + wet * 0.09
        t = np.arange(self.n) / SR
        out *= np.minimum(1, t / fade_in) * np.clip((self.n / SR - t) / (self.n / SR - fade_out_from), 0, 1)
        out = np.tanh(out * 1.1)
        return out / np.max(np.abs(out)) * 10 ** (-1 / 20)


# ---------- sounds ----------




def kick():
    t = tt(0.35)
    sweep = 48 * (1 + 3 * np.exp(-t / 0.03))
    return np.sin(2 * np.pi * np.cumsum(sweep) / SR) * env(t, 0.001, 0.14)


def hat():
    t = tt(0.06)
    return highpass(noise(0.06), 7000) * env(t, 0.0005, 0.012)




def bass(note, dur):
    t = tt(dur)
    f = hz(note)
    x = saw(f, t) + 0.5 * np.sin(2 * np.pi * f / 2 * t)
    return lowpass(x, 500 + 1800 * np.exp(-t / 0.05)) * env(t, 0.004, dur * 0.6)


def riser(dur):
    t = tt(dur)
    u = t / dur
    air = lowpass(highpass(noise(dur), 250 + 2500 * u ** 2), 600 + 9000 * u ** 2) * u ** 2.2
    f = 110 * 2 ** (u * 1.0)
    tone = lowpass(saw(f, t) + saw(f * 1.005, t), 400 + 3000 * u) * u ** 1.6
    return air * 0.9 + tone * 0.25


def crash():
    t = tt(2.2)
    return highpass(noise(2.2), 3500) * env(t, 0.002, 0.55)


def stab(root):
    """A bright major chord over three octaves, filter and amp both decaying."""
    t = tt(2.2)
    x = np.zeros_like(t)
    for n in (root - 12, root, root + 4, root + 7, root + 12, root + 16, root + 19):
        for d in (-0.08, 0.0, 0.08):
            x += saw(hz(n + d), t)
    x = lowpass(x / 21, 900 + 7000 * np.exp(-t / 0.18)) * env(t, 0.004, 0.75)
    sub = np.sin(2 * np.pi * hz(root - 24) * t) * env(t, 0.003, 0.5)
    return x * 1.6 + sub * 0.8




def thud():
    t = tt(0.6)
    sweep = 42 * (1 + 2.5 * np.exp(-t / 0.04))
    return np.sin(2 * np.pi * np.cumsum(sweep) / SR) * env(t, 0.001, 0.22) + 0.5 * lowpass(noise(0.6), 600) * env(t, 0.001, 0.06)


def pad(notes, dur):
    t = tt(dur)
    x = np.zeros_like(t)
    for k, n in enumerate(notes):
        for d in (-0.06, 0.06):
            f = hz(n + d)
            x += np.sin(2 * np.pi * f * t + k) + 0.25 * np.sin(2 * np.pi * 2 * f * t)
    x *= 0.85 + 0.15 * np.sin(2 * np.pi * 0.22 * t)
    a = np.minimum(1, t / 1.6)
    return lowpass(x / (2 * len(notes)), 1600) * a


# ---------- the score ----------
CHORDS = [(57, (0, 3, 7, 10, 14)), (53, (0, 4, 7, 11, 14)), (60, (0, 4, 7, 11, 14)), (55, (0, 4, 7, 10, 14))]  # Am9 Fmaj9 Cmaj9 G9


def score(ev):
    mix = Mix(ev["duration"])
    games, bench = ev["games"], ev["bench"]
    beat = 60 / 124
    for gi, g in enumerate(games):
        root, shape = CHORDS[gi % len(CHORDS)]
        notes = [root + n for n in shape]
        # The pad holds each game's chord from its start through the win, overlapping the next one.
        start = games[gi - 1]["landed"] - 0.3 if gi else 0.0
        stop = games[gi + 1]["start"] + 0.4 if gi + 1 < len(games) else bench["at"] + 0.6
        mix.add(start, pad(notes, stop - start) * np.clip((stop - start - tt(stop - start)) / 0.5, 0, 1), 0.12, verb=0.6)
        # A soft beat and the chord's root in the bass, from the start (or the landing) to the mate.
        b = games[gi - 1]["landed"] + 0.02 if gi else g["start"] - 0.05
        while b < g["mate"] - 0.02:
            mix.add(b, kick(), 0.34)
            mix.add(b + beat / 2, hat(), 0.05, pan=0.25)
            mix.add(b, bass(root - 24, beat * 0.45), 0.16)
            mix.add(b + beat / 2, bass(root - 24, beat * 0.45), 0.12)
            b += beat
        # Build into the mate, then one hit.
        rise = min(1.8, g["mate"] - g["start"] - 0.2)
        mix.add(g["mate"] - rise, riser(rise), 0.18, verb=0.2)
        mix.add(g["mate"], stab(root + 12), 0.34, verb=0.45)
        mix.add(g["mate"], crash(), 0.13, verb=0.4)
        if g["landed"]:
            mix.add(g["landed"], thud(), 0.3, verb=0.1)
    # The closer: the pad alone, settling on the last chord's home (C), under the leaderboard.
    at = bench["at"]
    mix.add(at - 0.4, pad([60, 67, 74, 76, 83], ev["duration"] - at + 0.5), 0.15, verb=0.8)
    mix.add(at - 0.4, pad([48, 55], ev["duration"] - at + 0.5), 0.06, verb=0.4)
    return mix.master(fade_in=0.15, fade_out_from=ev["duration"] - 0.6)


if __name__ == "__main__":
    src, out = sys.argv[1], sys.argv[2]
    ev = asyncio.run(cues())
    track = score(ev)
    with tempfile.TemporaryDirectory() as d:
        wav = os.path.join(d, "score.wav")
        wavfile.write(wav, SR, (track.T * 32767).astype(np.int16))
        # One static gain to -14 LUFS integrated (the usual target for web video), peaks held under -1.5 dBFS.
        meter = subprocess.run(["ffmpeg", "-hide_banner", "-i", wav, "-af", "ebur128", "-f", "null", "-"], capture_output=True, text=True).stderr
        loud = float(meter.rsplit("I:", 1)[1].split("LUFS")[0])
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", src, "-i", wav, "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                        "-af", f"volume={-14 - loud:.2f}dB,alimiter=limit=0.84:level=false", "-c:a", "aac", "-b:a", "256k", "-shortest", "-movflags", "+faststart", out],
                       check=True)
        if os.environ.get("KEEP_WAV"):
            subprocess.run(["cp", wav, os.environ["KEEP_WAV"]], check=True)
    print(f"{out}: scored ({ev['duration']} s)")
