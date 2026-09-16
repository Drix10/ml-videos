"""Fish: wander + flee. Spawns keep distance; eaten fish stay dead (countdown)."""
import numpy as np

import config


def spawn_pos(rng, shark_xy=None, min_dist=config.RESPAWN_MIN_DIST):
    p = rng.uniform([0, 0], [config.ARENA_W, config.ARENA_H])
    for _ in range(20):  # rejection-sample away from the shark
        if shark_xy is None or np.linalg.norm(p - np.asarray(shark_xy)) >= min_dist:
            return float(p[0]), float(p[1])
        p = rng.uniform([0, 0], [config.ARENA_W, config.ARENA_H])
    return float(p[0]), float(p[1])  # ponytail: crowded -> accept overlap


class Fish:
    def __init__(self, rng, shark_xy=None):
        self.x, self.y = spawn_pos(rng, shark_xy)
        self.vx, self.vy = rng.uniform(-1.5, 1.5, 2)
        self.alive = True

    def update(self, rng, shark_x, shark_y):
        if not self.alive:
            return
        W, H = config.ARENA_W, config.ARENA_H
        dx, dy = self.x - shark_x, self.y - shark_y
        dist = (dx * dx + dy * dy) ** 0.5
        if dist < 150:  # flee, stronger when close
            s = max(0, 1 - dist / 150) * 3 / max(dist, 1)
            self.vx += dx * s
            self.vy += dy * s
        self.vx += rng.uniform(-0.2, 0.2)  # wander
        self.vy += rng.uniform(-0.2, 0.2)
        sp = (self.vx ** 2 + self.vy ** 2) ** 0.5
        if sp > config.FISH_SPEED:  # speed cap
            self.vx, self.vy = self.vx / sp * config.FISH_SPEED, \
                self.vy / sp * config.FISH_SPEED
        self.x += self.vx
        self.y += self.vy
        if self.x < 0 or self.x > W:  # bounce off walls
            self.vx *= -1
            self.x = min(max(self.x, 0), W)
        if self.y < 0 or self.y > H:
            self.vy *= -1
            self.y = min(max(self.y, 0), H)
