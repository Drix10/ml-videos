"""Entry: evolve per level (blind -> full sense), replay best, level up on threshold."""
import csv
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pygame

import config
import visualizer
from environment import Arena, simulate
from evolution import next_generation
from model import Brain

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("logs", exist_ok=True)


def prune_checkpoints():
    files = sorted(glob.glob("checkpoints/best_*.pt"), key=os.path.getmtime)
    for f in files[:-config.MAX_CHECKPOINTS]:
        os.remove(f)


def draw_progress(screen, done, total, level_idx, gen):
    screen.fill(config.BG_COLOR)
    lvl = config.LEVELS[level_idx]
    t = visualizer.font(24, True).render(
        f"Level {level_idx + 1} \u00b7 {lvl['name']}  gen {gen}", True, config.PINK)
    screen.blit(t, t.get_rect(center=(config.WIDTH / 2, 300)))
    t2 = visualizer.font(20).render(f"evaluating {done}/{total}", True, config.TEXT_WHITE)
    screen.blit(t2, t2.get_rect(center=(config.WIDTH / 2, 340)))
    x0, bw = 70, config.WIDTH - 140
    pygame.draw.rect(screen, config.DIM_GRAY, (x0, 380, bw, 16), border_radius=8)
    pygame.draw.rect(screen, config.PINK, (x0, 380, bw * done / max(total, 1), 16),
                     border_radius=8)
    pygame.display.flip()


def level_up_population(pop, fitness, new_size):
    """Grow every brain (ranked, elites first) to the new input size."""
    order = np.argsort(fitness)[::-1]
    return [pop[i].grow(new_size) for i in
            [order[i % len(order)] for i in range(len(pop))]]


def main():
    pygame.init()
    pygame.display.set_caption("Shark Neuroevolution")
    screen = pygame.display.set_mode((config.WIDTH, config.HEIGHT))
    nx, ny, nw, nh = config.NETWORK_RECT
    gx, gy, _, _ = config.GAME_RECT
    panel = screen.subsurface((nx, ny, nw, nh))
    game = screen.subsurface((gx, gy, config.ARENA_W, config.ARENA_H))
    clock = pygame.time.Clock()
    fresh = not os.path.exists("logs/fitness.csv") or os.path.getsize("logs/fitness.csv") == 0
    log = open("logs/fitness.csv", "a", newline="")  # append: "w" wiped history
    wr = csv.writer(log)
    if fresh:
        wr.writerow(["level", "generation", "best", "mean", "worst"])

    level, gen = 0, 0
    pop = [Brain(len(config.LEVELS[0]["inputs"])) for _ in range(config.POPULATION_SIZE)]
    try:
        while True:
            n_in = len(config.LEVELS[level]["inputs"])
            fit = np.empty(len(pop))  # chunked eval: pump events, window stays alive
            for i, b in enumerate(pop):
                for e in pygame.event.get():
                    if e.type == pygame.QUIT:
                        return
                fit[i] = simulate(b, level)[0]
                if i % 5 == 0 or i == len(pop) - 1:
                    draw_progress(screen, i + 1, len(pop), level, gen)
            best = pop[int(np.argmax(fit))]
            wr.writerow([level + 1, gen, round(float(fit.max()), 1),
                         round(float(fit.mean()), 1), round(float(fit.min()), 1)])
            log.flush()
            best.save(f"checkpoints/best_L{level + 1}_gen{gen}.pt")
            prune_checkpoints()
            print(f"Level {level + 1} ({config.LEVELS[level]['name']}) | gen {gen} | "
                  f"best {fit.max():.1f} mean {fit.mean():.1f}", flush=True)
            pygame.display.set_caption(
                f"Shark Neuroevolution — L{level + 1} gen {gen} best {fit.max():.0f}")

            arena = Arena(best, level)  # rendered replay of best brain
            for _ in range(config.EPISODE_LENGTH):
                skip = False
                for e in pygame.event.get():
                    if e.type == pygame.QUIT:
                        return
                    if e.type == pygame.KEYDOWN and e.key in (pygame.K_SPACE, pygame.K_ESCAPE):
                        skip = True
                if skip or arena.step():
                    break
                visualizer.draw_network(panel, best, arena.obs, level)
                visualizer.draw_arena(game, arena)
                visualizer.draw_caption(screen, config.LEVELS[level]["caption"])
                pygame.display.flip()
                clock.tick(config.FPS)

            if level < len(config.LEVELS) - 1 and \
                    fit.max() >= config.LEVEL_THRESHOLDS[level]:
                level += 1  # grow brains, keep learned weights
                print(f"*** LEVEL UP -> Level {level + 1}: "
                      f"{config.LEVELS[level]['name']} ***", flush=True)
                pop = level_up_population(pop, fit, len(config.LEVELS[level]["inputs"]))
                gen = 0
                continue
            pop = next_generation(pop, fit)
            gen += 1
    finally:
        log.close()
        pygame.quit()


if __name__ == "__main__":
    main()
