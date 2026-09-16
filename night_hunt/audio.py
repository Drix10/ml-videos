"""Procedural blips: synthesized at runtime, no asset files.
Every call is safe with no audio hardware (headless/dummy drivers)."""
import numpy as np
import pygame

_ok = False
_catch = None
_up = None


def ensure():
    """Call BEFORE pygame.init() so pre_init sticks."""
    global _ok
    try:
        pygame.mixer.pre_init(44100, -16, 2)
        pygame.mixer.init()
        _ok = True
    except Exception:
        _ok = False


def _synth(parts, dur=0.2, vol=0.35):
    rate = 44100
    n = int(rate * dur)
    w = np.zeros(n)
    for f0, f1, delay, decay in parts:  # (start Hz, end Hz, delay s, decay)
        i = int(delay * rate)
        m = n - i
        if m <= 0:
            continue
        tt = np.arange(m) / rate
        f = np.linspace(f0, f1, m)
        w[i:] += np.sin(2 * np.pi * np.cumsum(f) / rate) * np.exp(-tt * decay)
    w /= max(1e-6, np.abs(w).max())
    stereo = np.stack([w, w], axis=1)
    return pygame.sndarray.make_sound((stereo * vol * 32767).astype(np.int16))


def catch():
    global _catch
    if not _ok:
        return
    try:
        if _catch is None:  # soft pluck: 520 -> 240 Hz
            _catch = _synth([(520, 240, 0, 30)], 0.12)
        _catch.play()
    except Exception:
        pass


def levelup():
    global _up
    if not _ok:
        return
    try:
        if _up is None:  # rising triad C-E-G
            _up = _synth([(392, 392, 0, 9), (494, 494, 0.09, 9),
                          (587, 587, 0.18, 9)], 0.5)
        _up.play()
    except Exception:
        pass
