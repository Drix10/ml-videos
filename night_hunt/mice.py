"""Mice: brain-steered prey. One shared brain per school (species brain)."""
import math

import numpy as np

import config


def spawn_pos(rng, owl_xy=None, min_dist=config.RESPAWN_MIN_DIST):
    m = config.ARENA_MARGIN
    p = rng.uniform([m, m], [config.ARENA_W - m, config.ARENA_H - m])
    for _ in range(20):  # rejection-sample away from the owl
        if owl_xy is None or np.linalg.norm(p - np.asarray(owl_xy)) >= min_dist:
            return float(p[0]), float(p[1])
        p = rng.uniform([m, m], [config.ARENA_W - m, config.ARENA_H - m])
    return float(p[0]), float(p[1])  # ponytail: crowded -> accept overlap


class Mouse:
    def __init__(self, rng, owl_xy=None):
        self.x, self.y = spawn_pos(rng, owl_xy)
        self.angle = float(rng.uniform(0, 2 * math.pi))
        self.alive = True
        self.survived = 0  # frames lived: the fitness that matters
        self.trail = []  # short motion ribbon (visual only)

    def get_inputs(self, owl, level_idx) -> np.ndarray:
        names = config.LEVELS[level_idx]["inputs"]
        W, H = config.ARENA_W, config.ARENA_H
        dx, dy = owl.x - self.x, owl.y - self.y
        nd = math.hypot(dx, dy)
        full = {
            "bias": 1.0,
            "dist": nd / max(W, H),
            "dir x": dx / max(nd, 1.0),
            "dir y": dy / max(nd, 1.0),
            "closing": (math.cos(self.angle) * dx + math.sin(self.angle) * dy)
                       / max(nd, 1.0),
            "wall \u2191": self.y / H,
            "wall \u2193": (H - self.y) / H,
            "wall \u2192": (W - self.x) / W,
            "wall \u2190": self.x / W,
            "aim x": owl.x / W - 0.5,
            "aim y": owl.y / H - 0.5,
        }
        return np.array([full[n] for n in names], dtype=np.float32)

    def update(self, turn: float):
        self.angle = (self.angle + turn * config.TURN_RATE) % (2 * math.pi)
        self.x += math.cos(self.angle) * config.MOUSE_SPEED
        self.y += math.sin(self.angle) * config.MOUSE_SPEED
        m = config.ARENA_MARGIN
        if self.x < m or self.x > config.ARENA_W - m:
            self.angle = math.pi - self.angle
            self.x = min(max(self.x, m), config.ARENA_W - m)
        if self.y < m or self.y > config.ARENA_H - m:
            self.angle = -self.angle
            self.y = min(max(self.y, m), config.ARENA_H - m)
        self.survived += 1
        self.trail.append((self.x, self.y))
        if len(self.trail) > 6:
            self.trail.pop(0)
