"""Entry: evolve headless (with progress bar), replay best shark rendered."""
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

CAPTIONS = ["Each fish represents a reward signal",
            "Pink = strong weight, gray = weak",
            f"Mutation rate: {config.MUTATION_RATE * 100:.0f}%  (SPACE skips replay)"]

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("logs", exist_ok=True)

_fonts = {}


def font(size):
    if size not in _fonts:  # same leak as before: never SysFont() per-frame
        _fonts[size] = pygame.font.SysFont(None, size)
    return _fonts[size]


def prune_checkpoints():
    files = sorted(glob.glob("checkpoints/best_gen_*.pt"),
                   key=os.path.getmtime)
    for f in files[:-config.MAX_CHECKPOINTS]:
        os.remove(f)


def draw_progress(screen, done, total, gen):
    screen.fill((8, 8, 14))
    t = font(28).render(f"Generation {gen}: evaluating {done}/{total}", True, (220, 220, 230))
    screen.blit(t, t.get_rect(center=(config.ARENA_W / 2, 200)))
    x0, bw = 150, config.ARENA_W - 300
    pygame.draw.rect(screen, (42, 42, 52), (x0, 250, bw, 18), border_radius=9)
    pygame.draw.rect(screen, (255, 70, 180), (x0, 250, bw * done / max(total, 1), 18),
                     border_radius=9)
    pygame.display.flip()


def draw_game(surf, arena, gen):
    surf.fill((8, 8, 14))
    if len(arena.trail) > 1:  # fading motion trail
        for i, (a, b) in enumerate(zip(list(arena.trail)[:-1], list(arena.trail)[1:])):
            alpha = i / max(len(arena.trail) - 1, 1)
            pygame.draw.line(surf, (40, 40 + int(60 * alpha), 70 + int(60 * alpha)), a, b, 3)
    for f in arena.fish:  # fish: small white ellipses
        pygame.draw.ellipse(surf, (220, 220, 220), (f.pos[0] - 5, f.pos[1] - 3, 10, 6))
    if arena.flash:  # eat burst: expanding pink ring
        r = config.EAT_RADIUS + (20 - arena.flash) * 2
        pygame.draw.circle(surf, (255, 70, 180), arena.flash_pos.astype(int), r, 2)
    s = arena.shark  # white triangle rotated to velocity
    ang = np.arctan2(-s.vel[1], s.vel[0] + 1e-6) if (s.vel != 0).any() else 0
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    pts = [np.array([16, 0]), np.array([-10, 9]), np.array([-10, -9])]
    pygame.draw.polygon(surf, (255, 255, 255), [s.pos + R @ p for p in pts])

    surf.blit(font(24).render(f"Level {gen}  blind", True, (200, 200, 200)), (10, 8))
    left = f"{config.NUM_FISH - arena.eaten} / {config.NUM_FISH} FISH LEFT"
    t = font(24).render(left, True, (200, 200, 200))
    surf.blit(t, (config.ARENA_W - t.get_width() - 10, 8))  # right-aligned, never clips
    pygame.draw.ellipse(surf, (255, 70, 180),  # pink fish icon (emoji = tofu boxes)
                        (config.ARENA_W - t.get_width() - 30, 12, 12, 7))
    surf.blit(font(24).render(f"fitness {s.fitness:.0f}   eaten {arena.eaten}",
                              True, (140, 140, 150)), (10, 34))  # LIVE replay fit
    cap = CAPTIONS[(pygame.time.get_ticks() // 3000) % len(CAPTIONS)]
    t = font(24).render(cap, True, (140, 140, 150))
    surf.blit(t, t.get_rect(center=(config.ARENA_W / 2, config.ARENA_H - 18)))


def main():
    pygame.init()
    pygame.display.set_caption("Shark Neuroevolution")
    screen = pygame.display.set_mode((config.ARENA_W, config.ARENA_H + config.PANEL_H))
    panel = screen.subsurface((0, 0, config.ARENA_W, config.PANEL_H))
    game = screen.subsurface((0, config.PANEL_H, config.ARENA_W, config.ARENA_H))
    clock = pygame.time.Clock()
    fresh = not os.path.exists("logs/fitness.csv") or os.path.getsize("logs/fitness.csv") == 0
    log = open("logs/fitness.csv", "a", newline="")  # append: "w" wiped history every run
    wr = csv.writer(log)
    if fresh:
        wr.writerow(["generation", "best", "mean", "worst"])

    pop = [Brain() for _ in range(config.POPULATION_SIZE)]
    gen = 0
    try:
        while True:
            fit = np.empty(len(pop))  # chunked eval: pump events so window stays alive
            for i, b in enumerate(pop):
                for e in pygame.event.get():
                    if e.type == pygame.QUIT:
                        return
                fit[i] = simulate(b)[0]
                if i % 5 == 0 or i == len(pop) - 1:
                    draw_progress(screen, i + 1, len(pop), gen)
            best = pop[int(np.argmax(fit))]
            wr.writerow([gen, fit.max(), fit.mean(), fit.min()]); log.flush()
            best.save(f"checkpoints/best_gen_{gen}.pt")
            prune_checkpoints()
            print(f"gen {gen}: best {fit.max():.1f} mean {fit.mean():.1f}", flush=True)
            pygame.display.set_caption(f"Shark Neuroevolution — gen {gen} best {fit.max():.0f}")
            arena = Arena(best)  # rendered replay of best brain
            skip = False
            for _ in range(config.EPISODE_LENGTH):
                for e in pygame.event.get():
                    if e.type == pygame.QUIT:
                        return
                    if e.type == pygame.KEYDOWN and e.key in (pygame.K_SPACE, pygame.K_ESCAPE):
                        skip = True  # 30s replay is long once it gets good
                if skip or arena.eaten >= config.NUM_FISH:
                    break
                arena.step()
                visualizer.draw(panel, best, arena.obs, arena.out)
                draw_game(game, arena, gen)
                pygame.display.flip()
                clock.tick(config.FPS)
            pop = next_generation(pop, fit)
            gen += 1
    finally:
        log.close()
        pygame.quit()


if __name__ == "__main__":
    main()
