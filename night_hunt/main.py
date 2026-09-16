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
MAX_GENS = int(os.environ.get("MAX_GENS", "0") or 0)  # 0 = endless
SKIP_REPLAY = bool(os.environ.get("SKIP_REPLAY", ""))  # headless monitor runs


def prune_checkpoints():
    files = sorted(glob.glob("checkpoints/best_*.pt"), key=os.path.getmtime)
    for f in files[:-config.MAX_CHECKPOINTS]:
        os.remove(f)


def draw_progress(screen, done, total, level_idx, gen):
    screen.fill(config.BG_COLOR)
    lvl = config.LEVELS[level_idx]
    t = visualizer.font(24, True).render(
        f"Level {level_idx + 1} \u00b7 {lvl['name']}  gen {gen}", True, config.ACCENT)
    screen.blit(t, t.get_rect(center=(config.WIDTH / 2, 300)))
    t2 = visualizer.font(20).render(f"evaluating {done}/{total}", True, config.TEXT_WHITE)
    screen.blit(t2, t2.get_rect(center=(config.WIDTH / 2, 340)))
    x0, bw = 70, config.WIDTH - 140
    pygame.draw.rect(screen, config.DIM_GRAY, (x0, 380, bw, 16), border_radius=8)
    pygame.draw.rect(screen, config.ACCENT, (x0, 380, bw * done / max(total, 1), 16),
                     border_radius=8)
    pygame.display.flip()


def level_up_population(pop, fitness, new_size):
    """Grow every brain (ranked, elites first) to the new input size."""
    order = np.argsort(fitness)[::-1]
    return [pop[i].grow(new_size) for i in
            [order[i % len(order)] for i in range(len(pop))]]


def main():
    pygame.init()
    pygame.display.set_caption("Night Hunt")
    screen = pygame.display.set_mode((config.WIDTH, config.HEIGHT))
    nx, ny, nw, nh = config.NETWORK_RECT
    gx, gy, _, _ = config.GAME_RECT
    panel = screen.subsurface((nx, ny, nw, nh))
    game = screen.subsurface((gx, gy, config.ARENA_W, config.ARENA_H))
    clock = pygame.time.Clock()
    HEADER = ["level", "generation", "best", "mean", "worst",
              "catch_best", "catch_mean"]
    log_path = "logs/fitness.csv"
    if os.path.exists(log_path) and os.path.getsize(log_path):
        with open(log_path) as f:  # schema changed? archive, start clean
            if f.readline().strip() != ",".join(HEADER):
                try:
                    os.rename(log_path,
                              f"logs/fitness_legacy_{int(__import__('time').time())}.csv")
                except PermissionError:  # live run holds it: use session file
                    stamp = int(__import__('time').time())
                    log_path = f"logs/fitness_run_{stamp}.csv"
                    print(f"[log] fitness.csv locked, writing {log_path}", flush=True)
    fresh = not os.path.exists(log_path) or os.path.getsize(log_path) == 0
    log = open(log_path, "a", newline="")  # append: "w" wiped history
    wr = csv.writer(log)
    if fresh:
        wr.writerow(HEADER)

    level, gen, total = 0, 0, 0
    pop = [Brain(len(config.LEVELS[0]["inputs"])) for _ in range(config.POPULATION_SIZE)]
    try:
        while True:
            fit = np.empty(len(pop)); ate = np.empty(len(pop))  # chunked eval
            for i, b in enumerate(pop):  # averaged: a luck spike can't dominate
                for e in pygame.event.get():
                    if e.type == pygame.QUIT:
                        return
                res = [simulate(b, level) for _ in range(config.EVAL_EPISODES)]
                fit[i] = sum(f for f, _ in res) / len(res)
                ate[i] = sum(n for _, n in res) / len(res)
                if i % 5 == 0 or i == len(pop) - 1:
                    draw_progress(screen, i + 1, len(pop), level, gen)
            bi = int(np.argmax(fit))
            best = pop[bi]
            wr.writerow([level + 1, gen, round(float(fit.max()), 1),
                         round(float(fit.mean()), 1), round(float(fit.min()), 1),
                         int(round(ate[bi])), round(float(ate.mean()), 1)])
            log.flush()
            best.save(f"checkpoints/best_L{level + 1}_gen{gen}.pt")
            prune_checkpoints()
            print(f"Level {level + 1} ({config.LEVELS[level]['name']}) | gen {gen} | "
                  f"best {fit.max():.1f}/{config.LEVELS[level]['threshold']:.0f} "
                  f"lost {ate[bi]:.0f}/{config.NUM_MICE} "
                  f"mean {fit.mean():.1f} (lost {ate.mean():.1f}) "
                  f"(gen {gen + 1}/{config.LEVELS[level]['min_gens']})",
                  flush=True)
            pygame.display.set_caption(
                f"Night Hunt — L{level + 1} gen {gen} best {fit.max():.0f}")

            arena = Arena(best, level)  # rendered replay of best school
            show_new = gen < 3  # highlight brand-new senses for first 3 replays
            prev_n = len(config.LEVELS[level - 1]["inputs"]) if level > 0 else 0
            new_n = len(config.LEVELS[level]["inputs"]) - prev_n
            for _ in range(0 if SKIP_REPLAY else config.EPISODE_LENGTH):
                skip = False
                for e in pygame.event.get():
                    if e.type == pygame.QUIT:
                        return
                    if e.type == pygame.KEYDOWN and e.key in (pygame.K_SPACE, pygame.K_ESCAPE):
                        skip = True
                if skip or arena.step():
                    break
                visualizer.draw_network(panel, best, arena.obs, level, new_n, show_new)
                visualizer.draw_arena(game, arena)
                visualizer.draw_caption(screen, config.LEVELS[level]["caption"])
                pygame.display.flip()
                clock.tick(config.FPS)

            lvl = config.LEVELS[level]
            total += 1  # every generation counts, level-up or not
            if MAX_GENS and total >= MAX_GENS:
                print(f"[monitor] capped at {total} generations", flush=True)
                return
            if level < len(config.LEVELS) - 1 and gen + 1 >= lvl["min_gens"] \
                    and fit.max() >= lvl["threshold"]:
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
