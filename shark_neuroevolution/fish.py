"""Fish: wander + flee from shark. Spawns keep distance so no free food."""
import numpy as np

import config


def spawn_pos(rng, shark_pos=None, min_dist=config.RESPAWN_MIN_DIST):
    p = rng.uniform([0, 0], [config.ARENA_W, config.ARENA_H])
    for _ in range(20):  # rejection-sample away from the shark
        if shark_pos is None or np.linalg.norm(p - shark_pos) >= min_dist:
            return p
        p = rng.uniform([0, 0], [config.ARENA_W, config.ARENA_H])
    return p  # ponytail: crowded corner -> accept overlap, next frame still fine


class Fish:
    def __init__(self, rng, shark_pos=None):
        self.pos = spawn_pos(rng, shark_pos)
        self.vel = rng.uniform(-1, 1, 2) * config.FISH_SPEED

    def step(self, rng, shark_pos):
        away = self.pos - shark_pos  # flee vector
        d = np.linalg.norm(away) + 1e-6
        flee = (away / d) * max(0, 1 - d / 150) * 3  # strong when close
        wander = rng.normal(0, 0.4, 2)
        self.vel = (self.vel * 0.9 + flee + wander)
        sp = np.linalg.norm(self.vel) + 1e-6
        self.vel *= min(sp, config.FISH_SPEED * 1.5) / sp
        self.pos = np.clip(self.pos + self.vel, 0, (config.ARENA_W, config.ARENA_H))
