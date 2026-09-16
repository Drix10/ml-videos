"""Shark: angle-steered agent. Senses depend on level (blind -> full sense)."""
import math

import numpy as np

import config
from model import Brain


class Agent:
    def __init__(self, brain, x=None, y=None):
        self.brain = brain or Brain(len(config.LEVELS[0]["inputs"]))
        self.x = config.ARENA_W / 2 if x is None else x
        self.y = config.ARENA_H / 2 if y is None else y
        self.angle = float(np.random.uniform(0, 2 * math.pi))
        self.fitness = 0.0
        self.trail = []

    def get_inputs(self, fish_list, level_idx) -> np.ndarray:
        names = config.LEVELS[level_idx]["inputs"]
        W, H = config.ARENA_W, config.ARENA_H
        # nearest fish (linear scan is fine: NUM_FISH is small)
        nd, ndx, ndy, nfx, nfy = float("inf"), 0.0, 0.0, 0.0, 0.0
        for f in fish_list:
            if not f.alive:
                continue
            dx, dy = f.x - self.x, f.y - self.y
            d = math.hypot(dx, dy)
            if d < nd:
                nd, ndx, ndy, nfx, nfy = d, dx, dy, f.x, f.y
        if nd == float("inf"):  # no fish left: neutral sensors
            nd, ndx, ndy = max(W, H), 0.0, 0.0
        full = {
            "bias": 1.0,
            "dist": nd / max(W, H),
            "dir x": ndx / max(nd, 1.0),
            "dir y": ndy / max(nd, 1.0),
            "closing": (math.cos(self.angle) * ndx + math.sin(self.angle) * ndy)
                       / max(nd, 1.0),
            "wall \u2191": self.y / H,
            "wall \u2193": (H - self.y) / H,
            "wall \u2192": (W - self.x) / W,
            "wall \u2190": self.x / W,
            "aim x": nfx / W - 0.5,
            "aim y": nfy / H - 0.5,
        }
        return np.array([full[n] for n in names], dtype=np.float32)

    def update(self, fish_list, level_idx) -> bool:
        """Sense -> steer -> move -> bounce. Returns True on wall bounce."""
        W, H = config.ARENA_W, config.ARENA_H
        obs = self.get_inputs(fish_list, level_idx)
        out = float(self.brain.act(obs)[0])
        self.angle = (self.angle + out * config.TURN_RATE) % (2 * math.pi)
        self.x += math.cos(self.angle) * config.SHARK_SPEED
        self.y += math.sin(self.angle) * config.SHARK_SPEED
        bounced = False
        if self.x < 0 or self.x > W:  # bounce: mirror angle, clamp inside
            self.angle = math.pi - self.angle
            self.x = min(max(self.x, 0), W)
            bounced = True
        if self.y < 0 or self.y > H:
            self.angle = -self.angle
            self.y = min(max(self.y, 0), H)
            bounced = True
        self.trail.append((self.x, self.y))
        if len(self.trail) > config.MAX_TRAIL:
            self.trail.pop(0)
        return bounced
