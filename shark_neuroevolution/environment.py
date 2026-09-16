"""Arena: headless simulate() for training + state for rendered replay."""
from collections import deque

import numpy as np

import config
from agent import Shark
from fish import Fish, spawn_pos


def _eat(shark, fish, rng):
    """Respawn eaten fish AWAY from shark; returns count eaten this step."""
    n = 0
    for f in fish:
        if np.linalg.norm(f.pos - shark.pos) < config.EAT_RADIUS:
            n += 1
            shark.fitness += config.EAT_REWARD
            f.pos = spawn_pos(rng, shark.pos)
            f.vel = rng.uniform(-1, 1, 2) * config.FISH_SPEED
    return n


def simulate(brain, rng=None, steps=config.EPISODE_LENGTH, n_fish=config.NUM_FISH):
    """Run one episode headless. Returns (fitness, fish_eaten)."""
    rng = rng or np.random.default_rng()
    shark = Shark(brain)
    fish = [Fish(rng, shark.pos) for _ in range(n_fish)]
    eaten = 0
    for _ in range(steps):
        if shark.step(shark.sense(fish)):  # wall penalty here too (was replay-only)
            shark.fitness -= config.WALL_PENALTY
        for f in fish:
            f.step(rng, shark.pos)
        eaten += _eat(shark, fish, rng)
        shark.fitness -= config.IDLE_PENALTY
    return shark.fitness, eaten


class Arena:  # live, renderable episode of one shark (for main.py replay)
    def __init__(self, brain, rng=None):
        self.rng = rng or np.random.default_rng()
        self.shark = Shark(brain)
        self.fish = [Fish(self.rng, self.shark.pos) for _ in range(config.NUM_FISH)]
        self.eaten, self.obs = 0, np.zeros(config.INPUT_NODES, dtype=np.float32)
        self.out = np.zeros(config.OUTPUT_NODES, dtype=np.float32)
        self.trail = deque(maxlen=config.TRAIL_LEN)
        self.flash, self.flash_pos = 0, np.zeros(2)  # eat-burst ring timer + pos

    def step(self):
        self.obs = self.shark.sense(self.fish)
        self.out = self.shark.brain.act(self.obs)  # cached for visualizer bars
        if self.shark.step(self.obs):
            self.shark.fitness -= config.WALL_PENALTY
        for f in self.fish:
            f.step(self.rng, self.shark.pos)
        n = _eat(self.shark, self.fish, self.rng)
        if n:  # trigger ring burst at shark pos
            self.eaten += n
            self.flash, self.flash_pos = 20, self.shark.pos.copy()
        self.shark.fitness -= config.IDLE_PENALTY
        self.trail.append(self.shark.pos.copy())
        self.flash = max(0, self.flash - 1)
