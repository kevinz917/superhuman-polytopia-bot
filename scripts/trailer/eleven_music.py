"""Music for the hands trailer from the ElevenLabs Music API: takes of a style, each muxed into a rendered video.

Each take is one continuous piece, composed as a plan in the trailer's shape: four 3 s rounds that build and land a
hit at their end (the mates fall at about 2.9, 6.1, 8.9 and 11.5 s), then a long closing section where the same piece
carries on softer and sustained under the leaderboard. The model winds a piece down over its last few seconds, so the
closing section runs CLOSING_MS (16 s, a 28 s piece) and the piece is cut at 20 s, leaving that ending after the
video; the cut gets a short fade. If the API rejects the plan, the same brief
goes as a plain prompt.

Usage (key in the environment, never on the command line):
  ELEVENLABS_API_KEY=... uv run --no-project python -I scripts/trailer/eleven_music.py IN.mp4 OUT_DIR STYLE [STYLE ...]
  TAKES=3 sets how many takes per style (default 3), FIRST_TAKE the first take's number (default 1); each take has its
  own seed and a flavour.
"""

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

API = "https://api.elevenlabs.io/v1/music?output_format=mp3_44100_192"
DURATION = 20.0
CLOSING_MS = int(os.environ.get("CLOSING_MS", 16000))

STYLES = {
    "epic": (["cinematic hybrid trailer", "pulsing synth ostinato", "taiko and low brass hits", "rising tension"],
             ["vocals", "lo-fi"],
             [[], ["heroic French horn melody"], ["darker, choir pads and cello ostinato"]]),
    "electronic": (["modern electronic trailer", "punchy sidechained synths", "future bass chord stabs", "tight drums"],
                   ["vocals", "orchestra"], [[], ["glassy arpeggios"], ["deep reese bass"]]),
    "strategy": (["playful orchestral strategy-game theme", "pizzicato strings", "bright brass stabs", "snare build-ups"],
                 ["vocals", "dubstep"],
                 [[], ["whimsical woodwinds and glockenspiel"], ["bold marching snare and horns"]]),
    "keynote": (["minimal tech keynote music", "clean plucked synth arpeggio", "soft kick pulse", "airy pads"],
                ["vocals", "heavy distortion"], [[], ["warm piano"], ["glassy bells"]]),
}


def plan(name, take):
    styles, avoid, flavours = STYLES[name]
    round_ = lambda k: {  # noqa: E731
        "section_name": f"Round {k}",
        "positive_local_styles": ["driving pulse", "builds quickly", f"lands a {'big ' if k < 4 else 'biggest '}hit at the end"],
        "negative_local_styles": ["fade out", "silence"],
        "duration_ms": 3000,
        "lines": [],
    }
    return {
        "positive_global_styles": styles + flavours[take % len(flavours)] + ["instrumental", "120-128 bpm", "one continuous piece"],
        "negative_global_styles": avoid,
        "sections": [round_(k) for k in range(1, 5)]
        + [{"section_name": "Closing",
            "positive_local_styles": ["the same instruments and theme carry on, softer", "sustained warm chords",
                                      "steady level to the end", "resolved"],
            "negative_local_styles": ["silence", "fade out", "stop", "new song", "big hits"],
            "duration_ms": CLOSING_MS, "lines": []}],
    }


def prompt(name, take):
    styles, avoid, flavours = STYLES[name]
    return (f"{12 + CLOSING_MS // 1000}-second instrumental trailer cue, one continuous piece: {', '.join(styles + flavours[take % len(flavours)])}. "
            "Four quick rounds: each builds for about three seconds and lands a hit (around 3, 6, 9 and 11.5 s), each "
            "bigger than the last; then from 12 s the same piece carries on softer, with sustained warm chords at a steady "
            f"level to the end. No {', no '.join(avoid)}.")


def compose(body, key):
    """POST a composition; retries busy (429), server errors and dropped connections with backoff."""
    for attempt in range(6):
        req = urllib.request.Request(API, data=json.dumps(body).encode(), method="POST",
                                     headers={"xi-api-key": key, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code != 429 and e.code < 500 or attempt == 5:
                raise
        except urllib.error.URLError:
            if attempt == 5:
                raise
        time.sleep(10 * 2**attempt)


def mux(video, audio, out):
    """Cut to the video's length with short fades, one static gain to -14 LUFS, peaks under -1.5 dBFS."""
    shape = f"atrim=0:{DURATION},afade=t=in:d=0.15,afade=t=out:st={DURATION - 0.6}:d=0.6"
    meter = subprocess.run(["ffmpeg", "-hide_banner", "-i", audio, "-af", f"{shape},ebur128", "-f", "null", "-"],
                           capture_output=True, text=True).stderr
    loud = float(meter.rsplit("I:", 1)[1].split("LUFS")[0])
    af = f"{shape},volume={-14 - loud:.2f}dB,alimiter=limit=0.84:level=false"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", video, "-i", audio, "-map", "0:v", "-map", "1:a",
                    "-c:v", "copy", "-af", af, "-c:a", "aac", "-b:a", "256k", "-shortest", "-movflags", "+faststart", out],
                   check=True)


if __name__ == "__main__":
    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        sys.exit("set ELEVENLABS_API_KEY")
    video, out_dir, names = sys.argv[1], sys.argv[2], sys.argv[3:]
    takes, first = int(os.environ.get("TAKES", 3)), int(os.environ.get("FIRST_TAKE", 1)) - 1
    os.makedirs(out_dir, exist_ok=True)
    for name in names:
        for take in range(first, first + takes):
            mp3 = os.path.join(out_dir, f"music-{name}-{take + 1}.mp3")
            if not os.path.exists(mp3):
                try:
                    audio = compose({"composition_plan": plan(name, take), "model_id": "music_v1", "seed": 101 + take,
                                     "respect_sections_durations": True}, key)
                except urllib.error.HTTPError as e:
                    print(f"{name} {take + 1}: plan rejected ({e.code} {e.read()[:200]!r}), sending as a prompt", flush=True)
                    audio = compose({"prompt": prompt(name, take), "music_length_ms": 12000 + CLOSING_MS, "force_instrumental": True,
                                     "model_id": "music_v1"}, key)
                open(mp3, "wb").write(audio)
            out = os.path.join(out_dir, f"trailer-{name}-{take + 1}.mp4")
            mux(video, mp3, out)
            print(f"{name} {take + 1}: {out}", flush=True)
