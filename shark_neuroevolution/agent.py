"""Shark: holds brain, senses nearest fish, steers."""
import numpy as np

import config
from model import Brain


class Shark:
    def __init__(self, brain=None):
        self.brain = brain or Brain()
        self.pos = np.array([config.ARENA_W / 2, config.ARENA_H / 2], float)
        self.vel = np.zeros(2)
        self.fitness = 0.0

    def sense(self, fish) -> np.ndarray:
        k = config.INPUT_NODES // 2  # one (dx, dy) pair per slot
        nearest = sorted((f.pos - self.pos for f in fish),
                         key=lambda v: v @ v)[:k]
        obs = np.zeros(config.INPUT_NODES, dtype=np.float32)
        for i, v in enumerate(nearest):  # dx/W in [-1, 1], no *2 (old code hit [-2, 2])
            obs[2 * i:2 * i + 2] = v / (config.ARENA_W, config.ARENA_H)
        return obs  # short fish lists / odd INPUT_NODES just leave zeros

    def step(self, obs):
        accel = self.brain.act(obs)
        self.vel = (self.vel * 0.9 + accel * 1.5)
        sp = np.linalg.norm(self.vel) + 1e-6
        self.vel *= min(sp, config.SHARK_SPEED) / sp
        self.pos += self.vel
        # wall hit must be tested BEFORE clip, else the condition is always False
        hit = bool((self.pos <= 0).any() or
                   (self.pos >= (config.ARENA_W, config.ARENA_H)).any())
        self.pos = np.clip(self.pos, 0, (config.ARENA_W, config.ARENA_H))
        return hit
