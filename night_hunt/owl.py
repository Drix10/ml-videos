"""Fixed threat: positional pursuit hunter. Never learns, never tires."""
import math

import config


class Owl:
    def __init__(self, x=None, y=None):
        self.x = config.ARENA_W / 2 if x is None else x
        self.y = config.ARENA_H / 2 if y is None else y
        self.angle = 0.0
        self.trail = []

    def update(self, mice):
        alive = [m for m in mice if m.alive]
        if alive:  # steer toward nearest mouse, turn-limited (wide arcs)
            tgt = min(alive, key=lambda m: (m.x - self.x) ** 2 + (m.y - self.y) ** 2)
            des = math.atan2(tgt.y - self.y, tgt.x - self.x)
            d = (des - self.angle + math.pi) % (2 * math.pi) - math.pi
            self.angle = (self.angle + max(-config.OWL_TURN,
                                          min(config.OWL_TURN, d))) % (2 * math.pi)
        self.x += math.cos(self.angle) * config.OWL_SPEED
        self.y += math.sin(self.angle) * config.OWL_SPEED
        m = 14  # body extends past center: bounce inside so it never clips edges
        if self.x < m or self.x > config.ARENA_W - m:
            self.angle = math.pi - self.angle
            self.x = min(max(self.x, m), config.ARENA_W - m)
        if self.y < m or self.y > config.ARENA_H - m:
            self.angle = -self.angle
            self.y = min(max(self.y, m), config.ARENA_H - m)
        self.trail.append((self.x, self.y))
        if len(self.trail) > config.MAX_TRAIL:
            self.trail.pop(0)