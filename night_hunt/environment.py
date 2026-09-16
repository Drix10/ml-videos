"""Arena: headless simulate() + live Arena. One brain drives the whole school."""
import math

import numpy as np

import config
from mice import Mouse
from owl import Owl


def _episode_step(brain, owl, mice, rng, level_idx):
    """Shared physics: owl hunts, live mice steer via ONE batched forward pass."""
    owl.update(mice)
    alive = [m for m in mice if m.alive]
    if alive:
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
    fit = sum(m.survived for m in mice) / max(len(mice), 1)
    if caught == 0:  # untouched school earns the bonus
        fit += config.SURVIVE_BONUS
    return fit


def simulate(brain, level_idx, rng=None, steps=config.EPISODE_LENGTH,
             n_mice=config.NUM_MICE):
    """Run one episode headless. Returns (fitness, mice_caught)."""
    rng = rng or np.random.default_rng()
    owl = Owl()
    mice = [Mouse(brain, rng, (owl.x, owl.y)) for _ in range(n_mice)]
    caught = 0
    for _ in range(steps):
        caught += _episode_step(brain, owl, mice, rng, level_idx)
        if caught >= n_mice:
            break
    return _fitness(mice, caught), caught


class Arena:  # live, renderable episode of one school (for main.py replay)
    def __init__(self, brain, level_idx, rng=None):
        self.rng = rng or np.random.default_rng()
        self.level_idx = level_idx
        self.brain = brain
        self.owl = Owl()
        self.mice = [Mouse(brain, self.rng, (self.owl.x, self.owl.y))
                     for _ in range(config.NUM_MICE)]
        n_in = len(config.LEVELS[level_idx]["inputs"])
        self.obs = np.zeros(n_in, dtype=np.float32)
        self.out = 0.0
        self.caught = 0
        self.flash = 0  # catch-burst ring timer

    def _focus(self):  # camera mouse: first survivor (diagram shows its brain state)
        for m in self.mice:
            if m.alive:
                return m
        return self.mice[0]

    def step(self):
        f = self._focus()
        self.obs = f.get_inputs(self.owl, self.level_idx)
        self.out = float(self.brain.act(self.obs)[0])
        n = _episode_step(self.brain, self.owl, self.mice, self.rng, self.level_idx)
        if n:
            self.caught += n
            self.flash = 20
        self.flash = max(0, self.flash - 1)
        return self.caught >= config.NUM_MICE
