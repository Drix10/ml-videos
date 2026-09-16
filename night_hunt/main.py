"""Entry: headless trainer (no window) + cinematic showcase player.

- `python main.py` -> trains all levels, console only. No window ever.
- `python main.py showcase [levels...]` -> the edited video: each level's
  all-time best replay with level-up cards in between. SPACE skips ahead.
"""
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
S = config.UI_SCALE  # device pixels per logical unit (2 = 1080x1920)
MAX_GENS = int(os.environ.get("MAX_GENS", "0") or 0)  # 0 = endless

def prune_checkpoints():
    import csv as _csv
    keep = set()  # each level's all-time best must survive for showcase
    try:
        with open("logs/fitness.csv") as f:
            best = {}
            for row in _csv.DictReader(f):
                lv, g = int(row["level"]), int(row["generation"])
                ft = float(row["best"])
                if lv not in best or ft > best[lv][1]:
                    best[lv] = (g, ft)
            keep = {os.path.normpath(f"checkpoints/best_L{lv}_gen{g}.pt")
                    for lv, (g, _) in best.items()}
    except FileNotFoundError:
        pass
    files = sorted(glob.glob("checkpoints/best_*.pt"), key=os.path.getmtime)
    for f in files[:max(0, len(files) - config.MAX_CHECKPOINTS)]:
        if os.path.normpath(f) not in keep:
            os.remove(f)


def level_up_population(pop, fitness, new_size):
    """Grow every brain (ranked, elites first) to the new input size."""
    order = np.argsort(fitness)[::-1]
    return [pop[i].grow(new_size) for i in
            [order[i % len(order)] for i in range(len(pop))]]


def show_card(screen, clock, lv):
    """Level-up interstitial: gold flash, then the new sense held full-screen."""
    flash = pygame.Surface(screen.get_size())
    flash.fill(config.ACCENT)
    flash.set_alpha(160)
    screen.blit(flash, (0, 0))
    pygame.display.flip()
    pygame.time.wait(120)
    lvl = config.LEVELS[lv]
    sense = " + ".join(lvl["inputs"][-2:] if len(lvl["inputs"]) > 1
                        else lvl["inputs"])
    tag = visualizer.font(13, True).render(f"LEVEL {lv + 1}", True,
                                           config.BG_COLOR)
    br = tag.get_rect(center=(screen.get_size()[0] / 2,
                              screen.get_size()[1] / 2 - 60 * S))
    name = visualizer.font(30, True).render(lvl["name"], True, config.ACCENT)
    nr = name.get_rect(center=(screen.get_size()[0] / 2,
                               screen.get_size()[1] / 2 - 20 * S))
    plus = visualizer.font(34, True).render(f"+ {sense.upper()}", True,
                                            config.TEXT_WHITE)
    pr = plus.get_rect(center=(screen.get_size()[0] / 2,
                               screen.get_size()[1] / 2 + 30 * S))
    cap = visualizer.font(18, True).render(lvl["caption"], True,
                                           config.TEXT_GRAY)
    cr = cap.get_rect(center=(screen.get_size()[0] / 2,
                              screen.get_size()[1] / 2 + 70 * S))
    frames = int(1.5 * config.FPS)  # edit point: hold, SPACE skips
    for _ in range(frames):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                return True
            if e.type == pygame.KEYDOWN and e.key in (pygame.K_SPACE,
                                                      pygame.K_ESCAPE):
                return False
        screen.fill(config.BG_COLOR)
        pygame.draw.rect(screen, config.ACCENT, br.inflate(20 * S, 8 * S),
                         border_radius=4 * S)
        screen.blit(tag, br)
        screen.blit(name, nr)
        screen.blit(plus, pr)
        screen.blit(cap, cr)
        pygame.display.flip()
        clock.tick(config.FPS)
    return False


def showcase(screen, panel, game, clock, only=()):
    """Train-headless workflow: replay each level's all-time best school once.
    only: optional level numbers (e.g. showcase 2 5)."""
    import csv as _csv
    best = {}  # level -> (gen, fit)
    try:
        with open("logs/fitness.csv") as f:
            for row in _csv.DictReader(f):
                lv, g, ft = int(row["level"]), int(row["generation"]), float(row["best"])
                if lv not in best or ft > best[lv][1]:
                    best[lv] = (g, ft)
    except FileNotFoundError:
        print("[showcase] no logs/fitness.csv yet — train first", flush=True)
        return
    played = False
    for lv in sorted(best):
        if only and lv not in only:
            continue
        if played and not only:  # level-up card stitches the story together
            if show_card(screen, clock, lv - 1):
                return
        played = True
        g, ft = best[lv]
        try:
            brain = Brain.load(f"checkpoints/best_L{lv}_gen{g}.pt",
                               len(config.LEVELS[lv - 1]["inputs"]))
        except FileNotFoundError:
            print(f"[showcase] L{lv} gen {g} pruned, skipping", flush=True)
            continue
        print(f"[showcase] Level {lv} gen {g} (fit {ft:.0f}) — SPACE for next", flush=True)
        arena = Arena(brain, lv - 1)
        prev_n = len(config.LEVELS[lv - 2]["inputs"]) if lv > 1 else 0
        new_n = len(config.LEVELS[lv - 1]["inputs"]) - prev_n
        for _ in range(config.EPISODE_LENGTH):
            nxt = False
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    return
                if e.type == pygame.KEYDOWN and e.key in (pygame.K_SPACE, pygame.K_ESCAPE):
                    nxt = True
            if nxt or arena.step():
                break
            visualizer.draw_network(panel, brain, arena.obs, lv - 1, new_n)
            visualizer.draw_arena(game, arena)
            visualizer.draw_caption(screen, config.LEVELS[lv - 1]["caption"])
            pygame.display.flip()
            clock.tick(config.FPS)


def _showcase_levels():
    """CLI: `showcase [levels...]`. Also honors SHOWCASE=1 (all levels)."""
    if len(sys.argv) > 1 and sys.argv[1] == "showcase":
        return True, {int(a) for a in sys.argv[2:] if a.isdigit()}
    if os.environ.get("SHOWCASE"):
        return True, set()
    return False, set()


def main():
    want_show, only = _showcase_levels()
    if want_show:  # the ONLY path that opens a window
        pygame.init()
        pygame.display.set_caption("Night Hunt")
        screen = pygame.display.set_mode((config.WIDTH * S, config.HEIGHT * S))
        R = lambda r: tuple(v * S for v in r)  # logical rect -> device rect
        nx, ny, nw, nh = R(config.NETWORK_RECT)
        gx, gy, _, _ = R(config.GAME_RECT)
        panel = screen.subsurface((nx, ny, nw, nh))
        game = screen.subsurface((gx, gy, config.ARENA_W * S,
                                  config.ARENA_H * S))
        clock = pygame.time.Clock()
        showcase(screen, panel, game, clock, only)  # e.g. showcase 2 5
        pygame.quit()
        return
    print("[train] headless: no window. Watch the console, "
          "record later with `python main.py showcase`.", flush=True)
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
        while True:  # headless: evaluate, log, evolve. No window, no events.
            fit = np.empty(len(pop)); ate = np.empty(len(pop))  # chunked eval
            for i, b in enumerate(pop):  # averaged: a luck spike can't dominate
                # common spawns: same layouts for every brain, so score gaps
                # come from policy, not spawn luck (this makes fitness heritable)
                res = [simulate(b, level,
                                rng=np.random.default_rng((level, gen, ep)))
                       for ep in range(config.EVAL_EPISODES)]
                fit[i] = sum(f for f, _ in res) / len(res)
                ate[i] = sum(n for _, n in res) / len(res)
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

            lvl = config.LEVELS[level]
            last = level == len(config.LEVELS) - 1
            leveling = not last and gen + 1 >= lvl["min_gens"] \
                and fit.max() >= lvl["threshold"]
            finale = last and gen + 1 >= lvl["min_gens"] \
                and fit.max() >= config.FINALE_TARGET
            total += 1  # every generation counts, level-up or not
            if MAX_GENS and total >= MAX_GENS:
                print(f"[monitor] capped at {total} generations", flush=True)
                return
            if finale:
                print(f"*** FINALE CLEARED ({fit.max():.0f} >= "
                      f"{config.FINALE_TARGET}) — run `python main.py showcase` "
                      f"to record the video ***", flush=True)
                return
            if leveling:  # grow brains, keep learned weights
                print(f"*** LEVEL UP -> Level {level + 2}: "
                      f"{config.LEVELS[level + 1]['name']} ***", flush=True)
                level += 1
                pop = level_up_population(pop, fit, len(config.LEVELS[level]["inputs"]))
                gen = 0
                continue
            pop = next_generation(pop, fit)
            gen += 1
    finally:
        log.close()  # headless: nothing to quit


if __name__ == "__main__":
    main()
