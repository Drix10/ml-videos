"""Arena: headless simulate() for training + live Arena for rendered replay."""
import math

import config
from agent import Agent
from fish import Fish


def _episode_step(shark, fish, rng, level_idx):
    """One shared physics step. Returns fish eaten (0/1+). Both simulate()
    and Arena use it, so training fitness == replay fitness."""
    if shark.update(fish, level_idx):
        shark.fitness -= config.WALL_PENALTY
    for f in fish:
        f.update(rng, shark.x, shark.y)
    n = 0
    for f in fish:
        if f.alive and math.hypot(shark.x - f.x, shark.y - f.y) \
                < config.SHARK_RADIUS + config.FISH_RADIUS:
            f.alive = False
            n += 1
            shark.fitness += config.EAT_REWARD
    shark.fitness -= config.IDLE_PENALTY
    return n


def simulate(brain, level_idx, rng=None, steps=config.EPISODE_LENGTH,
             n_fish=config.NUM_FISH):
    """Run one episode headless. Returns (fitness, fish_eaten)."""
    rng = rng or __import__("numpy").random.default_rng()
    shark = Agent(brain)
    fish = [Fish(rng, (shark.x, shark.y)) for _ in range(n_fish)]
    eaten = 0
    for _ in range(steps):
        eaten += _episode_step(shark, fish, rng, level_idx)
        if eaten >= n_fish:
            shark.fitness += config.TIME_BONUS
            break
    return shark.fitness, eaten


class Arena:  # live, renderable episode of one shark (for main.py replay)
    def __init__(self, brain, level_idx, rng=None):
        import numpy as np
        self.rng = rng or np.random.default_rng()
        self.level_idx = level_idx
        self.shark = Agent(brain)
        self.fish = [Fish(self.rng, (self.shark.x, self.shark.y))
                     for _ in range(config.NUM_FISH)]
        self.eaten, self.obs = 0, np.zeros(
            len(config.LEVELS[level_idx]["inputs"]), dtype=np.float32)
        self.out = 0.0
        self.flash = 0  # eat-burst ring timer

    def step(self):
        self.obs = self.shark.get_inputs(self.fish, self.level_idx)
        self.out = float(self.shark.brain.act(self.obs)[0])
        n = _episode_step(self.shark, self.fish, self.rng, self.level_idx)
        if n:
            self.eaten += n
            self.flash = 20
        self.flash = max(0, self.flash - 1)
        return self.eaten >= config.NUM_FISH
