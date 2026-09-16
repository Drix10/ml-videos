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

    def get_inputs(self, owl, level_idx, out: np.ndarray = None) -> np.ndarray:
        """Perf: only compute the features this level actually uses (profiling
        showed the old build-a-dict-of-11-then-filter approach cost MORE per
        step than the neural net forward pass itself). `out`, if given, is a
        pre-allocated row (e.g. a view into a shared (n_mice, n_in) buffer)
        that this method fills in place instead of allocating a fresh array —
        lets the caller avoid one allocation per mouse per step.
        Bit-identical to the old dict-based version for every level (verified
        by fuzz test against the original implementation)."""
        names = config.LEVELS[level_idx]["inputs"]
        if out is None:
            out = np.empty(len(names), dtype=np.float32)
        W, H = config.ARENA_W, config.ARENA_H
        dx = owl.x - self.x
        dy = owl.y - self.y
        nd = math.hypot(dx, dy)
        nd_safe = max(nd, 1.0)
        for i, n in enumerate(names):
            if n == "bias":
                out[i] = 1.0
            elif n == "dist":
                out[i] = nd / max(W, H)
            elif n == "dir x":
                out[i] = dx / nd_safe
            elif n == "dir y":
                out[i] = dy / nd_safe
            elif n == "closing":
                out[i] = (math.cos(self.angle) * dx
                          + math.sin(self.angle) * dy) / nd_safe
            elif n == "wall \u2191":
                out[i] = self.y / H
            elif n == "wall \u2193":
                out[i] = (H - self.y) / H
            elif n == "wall \u2192":
                out[i] = (W - self.x) / W
            elif n == "wall \u2190":
                out[i] = self.x / W
            elif n == "aim x":
                out[i] = owl.x / W - 0.5
            elif n == "aim y":
                out[i] = owl.y / H - 0.5
        return out

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