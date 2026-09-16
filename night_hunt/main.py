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
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # logs/ land here
import numpy as np
import pygame
import torch

import config
import visualizer
from environment import Arena, simulate
from evolution import next_generation
from model import Brain

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("logs", exist_ok=True)
LOGD = os.environ.get("LOG_DIR", "logs")  # smoke tests point these at /tmp
CKPTD = os.environ.get("CKPT_DIR", "checkpoints")
os.makedirs(LOGD, exist_ok=True)
os.makedirs(CKPTD, exist_ok=True)
S = config.UI_SCALE  # device pixels per logical unit (2 = 1080x1920)
MAX_GENS = int(os.environ.get("MAX_GENS", "0") or 0)  # 0 = endless
RESUME = bool(os.environ.get("RESUME", ""))  # pick up from checkpoints/resume.pt

# Every step is ONE tiny forward pass (<=11 in, 9 hidden, 1 out). On a
# multi-core box PyTorch's default intra-op thread pool spends more time
# synchronizing threads than doing the actual matmul at this size — a single
# generation was measurably slower with the default thread count than with
# threads=1 in profiling. Pin to 1 thread; parallelism should come from
# running more episodes, not from parallelizing one episode's tiny math.
torch.set_num_threads(1)

SEED = os.environ.get("SEED")  # optional: set for a fully reproducible run
if SEED is not None:
    import random
    random.seed(int(SEED))
    np.random.seed(int(SEED))  # also seeds evolution.py's tournament draws
    torch.manual_seed(int(SEED))  # also seeds crossover/mutate's draws
    print(f"[seed] fixed at {SEED} — run is fully reproducible", flush=True)

def prune_checkpoints():
    import csv as _csv
    keep = set()  # each level's all-time best must survive for showcase
    try:
        with open(f"{LOGD}/fitness.csv") as f:
            best = {}
            for row in _csv.DictReader(f):
                lv, g = int(row["level"]), int(row["generation"])
                ft = float(row["best"])
                if lv not in best or ft > best[lv][1]:
                    best[lv] = (g, ft)
            keep = {os.path.normpath(f"{CKPTD}/best_L{lv}_gen{g}.pt")
                    for lv, (g, _) in best.items()}
    except FileNotFoundError:
        pass
    files = sorted(glob.glob(f"{CKPTD}/best_*.pt"), key=os.path.getmtime)
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
    prev = config.LEVELS[lv - 1]["inputs"] if lv > 0 else []
    sense = " + ".join(lvl["inputs"][len(prev):])  # only the NEW senses
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
        with open(f"{LOGD}/fitness.csv") as f:
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
            brain = Brain.load(f"{CKPTD}/best_L{lv}_gen{g}.pt",
                               len(config.LEVELS[lv - 1]["inputs"]))
        except (FileNotFoundError, RuntimeError) as ex:
            print(f"[showcase] L{lv} gen {g} unreadable ({ex}), skipping",
                  flush=True)
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
    log_path = f"{LOGD}/fitness.csv"
    if os.path.exists(log_path) and os.path.getsize(log_path):
        with open(log_path) as f:  # schema changed? archive, start clean
            header_ok = f.readline().strip() == ",".join(HEADER)
        # rename AFTER the read handle above is closed: on Windows, renaming
        # a file while still holding it open (even just for reading) can
        # raise PermissionError all on its own, with no other process
        # involved — that used to make a single, ordinary run mistake itself
        # for a locked-by-another-instance conflict every time the schema
        # changed.
        if not header_ok:
            try:
                os.rename(log_path,
                          f"{LOGD}/fitness_legacy_{int(__import__('time').time())}.csv")
            except PermissionError:  # NOW genuinely means another process
                stamp = int(__import__('time').time())          # holds it
                log_path = f"{LOGD}/fitness_run_{stamp}.csv"
                print(f"[log] fitness.csv locked, writing {log_path}", flush=True)
    fresh = not os.path.exists(log_path) or os.path.getsize(log_path) == 0
    log = open(log_path, "a", newline="")  # append: "w" wiped history
    wr = csv.writer(log)
    if fresh:
        wr.writerow(HEADER)

    level, gen, total = 0, 0, 0
    pop = [Brain(len(config.LEVELS[0]["inputs"])) for _ in range(config.POPULATION_SIZE)]
    bars = {}  # level -> current bar (autobar may lower a wall, loudly)
    hist = []  # recent gen-bests at this level (autobar's evidence)
    snap_path = f"{CKPTD}/resume.pt"
    if RESUME and os.path.exists(snap_path):
        snap = torch.load(snap_path, map_location="cpu", weights_only=True)
        if len(snap["pop"]) != config.POPULATION_SIZE:
            print("[resume] population size changed, starting fresh", flush=True)
        else:
            level, gen, total = snap["level"], snap["gen"], snap["total"]
            pop = [Brain(len(config.LEVELS[level]["inputs"]))
                   for _ in range(config.POPULATION_SIZE)]
            for b, sd in zip(pop, snap["pop"]):
                b.load_state_dict(sd)
            bars = snap.get("bars", {})
            print(f"[resume] Level {level + 1} gen {gen} "
                  f"({total} gens so far)", flush=True)
    elif RESUME:
        print(f"[resume] no checkpoint at {snap_path}, starting fresh",
              flush=True)
    def save_snapshot():
        # Always called before every `return` in the loop below, so a killed
        # (or MAX_GENS-capped) night really does lose nothing — see the
        # MAX_GENS fix note further down for why this used to be false.
        torch.save({"level": level, "gen": gen, "total": total,
                    "pop": [b.state_dict() for b in pop], "bars": bars},
                   snap_path)

    try:
        while True:  # headless: evaluate, log, evolve. No window, no events.
            fit = np.empty(len(pop)); ate = np.empty(len(pop))  # chunked eval
            for i, b in enumerate(pop):  # averaged: a luck spike can't dominate
                # Fixed per level (NOT per generation): every generation at
                # this level is scored on the exact same EVAL_EPISODES
                # layouts. Seeding by (level, gen, ep) instead — as this used
                # to do — meant an unchanged elite got re-scored on a fresh
                # random layout every generation, so fit.max() could swing
                # from pure spawn luck even with zero learning. That broke
                # three things at once: elitism no longer guaranteed
                # non-decreasing best fitness within a level, the "best"
                # checkpoint later loaded by showcase could just be the
                # luckiest seed rather than the best brain, and autobar's
                # stagnation read (median of recent bests) was measuring
                # seed noise as much as genuine plateauing.
                res = [simulate(b, level,
                                rng=np.random.default_rng((level, ep)))
                       for ep in range(config.EVAL_EPISODES)]
                fit[i] = sum(f for f, _ in res) / len(res)
                ate[i] = sum(n for _, n in res) / len(res)
            bi = int(np.argmax(fit))
            best = pop[bi]
            lvl = config.LEVELS[level]
            last = level == len(config.LEVELS) - 1
            bar = bars.setdefault(level, lvl["threshold"] if not last
                                  else config.FINALE_TARGET)
            hist.append(float(fit.max()))
            hist = hist[-10:]
            wr.writerow([level + 1, gen, round(float(fit.max()), 1),
                         round(float(fit.mean()), 1), round(float(fit.min()), 1),
                         int(round(ate[bi])), round(float(ate.mean()), 1)])
            log.flush()
            best.save(f"{CKPTD}/best_L{level + 1}_gen{gen}.pt")
            prune_checkpoints()
            print(f"Level {level + 1} ({config.LEVELS[level]['name']}) | gen {gen} | "
                  f"best {fit.max():.1f}/{bar:.0f} "
                  f"lost {ate[bi]:.0f}/{config.NUM_MICE} "
                  f"mean {fit.mean():.1f} (lost {ate.mean():.1f}) "
                  f"(gen {gen + 1}/{config.LEVELS[level]['min_gens']})",
                  flush=True)

            leveling = not last and gen + 1 >= lvl["min_gens"] \
                and fit.max() >= bar
            finale = last and gen + 1 >= lvl["min_gens"] \
                and fit.max() >= bar
            if not (leveling or finale) and gen + 1 >= 30 and len(hist) == 10:
                # autobar: a wall wastes nights. The median of recent bests
                # recurs ~every other gen, so this bar ALWAYS falls soon.
                new_bar = min(bar, float(np.median(hist)) + 10)
                if new_bar < bar - 1:
                    print(f"[autobar] L{level + 1} bar {bar:.0f} -> "
                          f"{new_bar:.0f} after {gen + 1} stuck gens",
                          flush=True)
                    bars[level] = bar = new_bar
                    # Re-derive with the SAME gate as above (min_gens included).
                    # Right now every level's min_gens (<=10) is well under
                    # the 30-gen floor that unlocks autobar, so dropping the
                    # check here was harmless in practice — but silently
                    # relying on that made it a landmine for the next config
                    # edit. Keep the gate explicit.
                    leveling = not last and gen + 1 >= lvl["min_gens"] \
                        and fit.max() >= bar
                    finale = last and gen + 1 >= lvl["min_gens"] \
                        and fit.max() >= bar
            total += 1  # every generation counts, level-up or not

            # NOTE on ordering, all three branches below: MAX_GENS/finale
            # used to be checked (and `return` immediately) BEFORE ever
            # reaching the snapshot save that used to sit at the bottom of
            # the loop, so stopping a run via MAX_GENS silently discarded
            # the very generation that had just been evaluated — resume.pt
            # still pointed at the previous generation. Every "train N gens
            # tonight" session was quietly losing its last generation of
            # progress. Saving a snapshot before every return fixes it;
            # worst case now is re-evaluating one already-known generation
            # on the next resume, not losing one.
            if leveling:  # grow brains, keep learned weights
                print(f"*** LEVEL UP -> Level {level + 2}: "
                      f"{config.LEVELS[level + 1]['name']} ***", flush=True)
                level += 1
                pop = level_up_population(pop, fit, len(config.LEVELS[level]["inputs"]))
                gen = 0
                hist = []
                save_snapshot()
                if MAX_GENS and total >= MAX_GENS:
                    print(f"[monitor] capped at {total} generations", flush=True)
                    return
                continue

            if finale:
                # Save the CURRENT (not re-evolved) population: there's
                # nothing to gain from spending a generation's worth of
                # crossover/mutate on a population we're about to discard.
                # A later `RESUME` picks up here and can keep polishing
                # the finale level if you want to push fit.max() higher.
                save_snapshot()
                print(f"*** FINALE CLEARED ({fit.max():.0f} >= "
                      f"{bar:.0f}) — run `python main.py showcase` "
                      f"to record the video ***", flush=True)
                return

            pop = next_generation(pop, fit)
            gen += 1
            save_snapshot()  # resume.pt: a killed (or capped) night loses nothing
            if MAX_GENS and total >= MAX_GENS:
                print(f"[monitor] capped at {total} generations", flush=True)
                return
    finally:
        log.close()  # headless: nothing to quit


if __name__ == "__main__":
    main()