"""Arena: headless simulate() + live Arena. One brain drives the whole school."""
import math

import numpy as np

import config
from mice import Mouse
from owl import Owl


class Spark:  # catch-burst mote: flies out, fades fast
    __slots__ = ("x", "y", "vx", "vy", "life")

    def __init__(self, x, y, rng):
        a = rng.uniform(0, 2 * math.pi)
        s = rng.uniform(1.5, 4.5)
        self.x, self.y = x, y
        self.vx, self.vy = math.cos(a) * s, math.sin(a) * s
        self.life = 1.0

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vx *= 0.94
        self.vy *= 0.94
        self.life -= 0.06


def _episode_step(brain, owl, mice, level_idx, obs_buf=None):
    """Shared physics: owl hunts, live mice steer via ONE batched forward pass.
    obs_buf: optional pre-allocated (n_mice, n_inputs) scratch array. When
    given, per-mouse observations are written into its rows in place instead
    of each mouse allocating (and get_inputs building) a fresh small array
    every step, which profiling showed was a bigger cost than the net itself."""
    owl.update(mice)
    alive = [m for m in mice if m.alive]
    if alive:
        if obs_buf is not None:
            obs = obs_buf[:len(alive)]
            for i, m in enumerate(alive):
                m.get_inputs(owl, level_idx, out=obs[i])
        else:
            obs = np.stack([m.get_inputs(owl, level_idx) for m in alive])
        turns = brain.act_batch(obs)
        for m, t in zip(alive, turns):
            m.update(float(t))
    n = 0
    for m in alive:
        if math.hypot(owl.x - m.x, owl.y - m.y) < config.CATCH_RADIUS:
            m.alive = False
            n += 1
    return n


def _fitness(mice, caught):
    # Deaths dominate, time breaks ties: one extra survivor (+1000) always
    # beats any survival-time gain (max ~1800). The GA must save mice first
    # and keep them alive longer second — that order IS the video's story.
    return ((len(mice) - caught) * config.SURVIVOR_WEIGHT
            + sum(m.survived for m in mice) / max(len(mice), 1))


def simulate(brain, level_idx, rng=None, steps=config.EPISODE_LENGTH,
             n_mice=config.NUM_MICE):
    """Run one episode headless. Returns (fitness, mice_caught)."""
    rng = rng or np.random.default_rng()
    owl = Owl()
    mice = [Mouse(rng, (owl.x, owl.y)) for _ in range(n_mice)]
    n_in = len(config.LEVELS[level_idx]["inputs"])
    obs_buf = np.empty((n_mice, n_in), dtype=np.float32)  # reused every step
    caught = 0
    for _ in range(steps):
        caught += _episode_step(brain, owl, mice, level_idx, obs_buf)
        if caught >= n_mice:
            break
    return _fitness(mice, caught), caught


class Arena:  # live, renderable episode of one school (showcase replays)
    def __init__(self, brain, level_idx, rng=None):
        self.rng = rng or np.random.default_rng()
        self.level_idx = level_idx
        self.brain = brain
        self.owl = Owl()
        self.mice = [Mouse(self.rng, (self.owl.x, self.owl.y))
                     for _ in range(config.NUM_MICE)]
        n_in = len(config.LEVELS[level_idx]["inputs"])
        self.obs = np.zeros(n_in, dtype=np.float32)
        self.obs_buf = np.empty((len(self.mice), n_in), dtype=np.float32)
        self.caught = 0
        self.flash = 0  # catch-burst ring timer
        self.sparks = []  # live particles
        self._last_focus = self.mice[0] if self.mice else None

    def _focus(self):
        """Camera mouse for the brain panel: first living survivor. If the
        whole school just got wiped, keep showing the last mouse that WAS
        alive (frozen at its final inputs) instead of jump-cutting to
        mice[0], which may be an arbitrary, unrelated mouse."""
        for m in self.mice:
            if m.alive:
                self._last_focus = m
                return m
        return self._last_focus

    def step(self):
        f = self._focus()
        if f is not None:
            self.obs = f.get_inputs(self.owl, self.level_idx)
        n = _episode_step(self.brain, self.owl, self.mice, self.level_idx,
                          self.obs_buf)
        if n:
            self.caught += n
            self.flash = 20
            self.sparks += [Spark(self.owl.x, self.owl.y, self.rng)
                            for _ in range(10)]
        self.flash = max(0, self.flash - 1)
        for s in self.sparks:
            s.update()
        self.sparks = [s for s in self.sparks if s.life > 0]
        return self.caught >= config.NUM_MICE