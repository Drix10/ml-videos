"""Fish: wander + flee. Spawns keep distance; eaten fish stay dead (countdown)."""
import numpy as np

import config


def spawn_pos(rng, shark_xy=None, min_dist=config.RESPAWN_MIN_DIST):
    m = config.FISH_MARGIN
    p = rng.uniform([m, m], [config.ARENA_W - m, config.ARENA_H - m])
    for _ in range(20):  # rejection-sample away from the shark
        if shark_xy is None or np.linalg.norm(p - np.asarray(shark_xy)) >= min_dist:
            return float(p[0]), float(p[1])
        p = rng.uniform([m, m], [config.ARENA_W - m, config.ARENA_H - m])
    return float(p[0]), float(p[1])  # ponytail: crowded -> accept overlap


class Fish:
    def __init__(self, rng, shark_xy=None):
        self.x, self.y = spawn_pos(rng, shark_xy)
        self.vx, self.vy = rng.uniform(-1.5, 1.5, 2)
        self.angle = 0.0
        self.alive = True

    def update(self, rng, shark_x, shark_y):
        if not self.alive:
            return
        W, H = config.ARENA_W, config.ARENA_H
        dx, dy = self.x - shark_x, self.y - shark_y
        dist = (dx * dx + dy * dy) ** 0.5
        if dist < config.FLEE_RADIUS:  # flee, stronger when close
            s = max(0, 1 - dist / config.FLEE_RADIUS) * config.FLEE_STRENGTH / max(dist, 1)
            self.vx += dx * s
            self.vy += dy * s
        self.vx += rng.uniform(-0.2, 0.2)  # wander
        self.vy += rng.uniform(-0.2, 0.2)
        sp = (self.vx ** 2 + self.vy ** 2) ** 0.5
        if sp > 0.3:  # heading for orienting the minnow silhouette
            import math
            self.angle = math.atan2(self.vy, self.vx)
        if sp > config.FISH_SPEED:  # speed cap
            self.vx, self.vy = self.vx / sp * config.FISH_SPEED, \
                self.vy / sp * config.FISH_SPEED
        self.x += self.vx
        self.y += self.vy
        m = config.FISH_MARGIN
        if self.x < m or self.x > W - m:  # bounce inside margin: never half-clipped
            self.vx *= -1
            self.x = min(max(self.x, m), W - m)
        if self.y < m or self.y > H - m:
            self.vy *= -1
            self.y = min(max(self.y, m), H - m)
