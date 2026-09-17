"""Genetic algorithm: elites + tournament + crossover + mutate."""
import numpy as np

import config
from model import Brain


def _tournament(pop, fitness):
    cands = np.random.choice(len(pop), min(3, len(pop)), replace=False)
    return pop[max(cands, key=lambda i: fitness[i])]


def next_generation(pop, fitness, mutation_rate=None, mutation_strength=None,
                     immigrant_frac=0.0):
    """mutation_rate/strength: override config's defaults for this call only
    -- used by main.py's stagnation shake to temporarily widen the search.
    immigrant_frac: fraction of the (non-elite) population replaced with
    brand-new random brains instead of crossover children, for the same
    reason -- reintroduce genetic diversity a converged population has lost.
    Elites are never touched by either knob: nothing already proven is ever
    put at risk by a shake."""
    rate = config.MUTATION_RATE if mutation_rate is None else mutation_rate
    strength = (config.MUTATION_STRENGTH if mutation_strength is None
                else mutation_strength)
    n, n_elite = len(pop), max(1, int(len(pop) * config.ELITE_FRACTION))
    order = np.argsort(fitness)[::-1]
    new = [pop[i] for i in order[:n_elite]]  # elites, unchanged
    n_immigrants = int(round(immigrant_frac * n)) if immigrant_frac else 0
    in_features = pop[0].fc1.in_features
    while len(new) < n:
        if n_immigrants > 0 and len(new) < n_elite + n_immigrants:
            new.append(Brain(in_features))  # fresh random genotype, no ancestry
            continue
        a, b = _tournament(pop, fitness), _tournament(pop, fitness)
        if len(pop) > 1:  # avoid cloning one parent with itself (was possible)
            for _ in range(5):
                if b is not a:
                    break
                b = _tournament(pop, fitness)
        child = Brain.crossover(a, b)
        child.mutate(rate=rate, strength=strength)
        new.append(child)
    return new