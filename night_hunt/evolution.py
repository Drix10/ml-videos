"""Genetic algorithm: elites + tournament + crossover + mutate."""
import numpy as np

import config
from model import Brain


def _tournament(pop, fitness):
    cands = np.random.choice(len(pop), min(3, len(pop)), replace=False)
    return pop[max(cands, key=lambda i: fitness[i])]


def next_generation(pop, fitness):
    n, n_elite = len(pop), max(1, int(len(pop) * config.ELITE_FRACTION))
    order = np.argsort(fitness)[::-1]
    new = [pop[i] for i in order[:n_elite]]  # elites, unchanged
    while len(new) < n:
        a, b = _tournament(pop, fitness), _tournament(pop, fitness)
        if len(pop) > 1:  # avoid cloning one parent with itself (was possible)
            for _ in range(5):
                if b is not a:
                    break
                b = _tournament(pop, fitness)
        child = Brain.crossover(a, b)
        child.mutate()
        new.append(child)
    return new