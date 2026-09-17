# Night hunt: fixed owl vs evolving mice. Portrait UI (phone/Reel-ready).
WIDTH, HEIGHT = 540, 960  # exact 9:16 for Reels (arena physics untouched)
FPS = 60
UI_SCALE = 1  # 1 = 540x960 watch window; 2 = crisp 1080x1920 recording

# Layout
NETWORK_RECT = (20, 44, 500, 380)   # brain diagram panel
GAME_RECT = (20, 444, 500, 380)     # arena panel (arena coords are 500x380)
ARENA_W, ARENA_H = 500, 380

# Colors: gold-on-midnight (not pink-on-black)
BG_COLOR = (10, 13, 26)
ARENA_COLOR = (15, 20, 38)
ARENA_BORDER = (110, 120, 165)  # bright rounded outline around the arena
OWL_COLOR = (245, 247, 255)
MOUSE_COLOR = (228, 234, 248)
ACCENT = (255, 182, 64)       # learned / strong / new
DIM_GRAY = (74, 78, 104)      # weak / inactive
TEXT_WHITE = (232, 236, 248)
TEXT_GRAY = (130, 138, 170)

# Game
NUM_MICE = 32
OWL_RADIUS = 16
CATCH_RADIUS = 7   # small: only a true hit counts, close shaves escape —
                    # this is what lets skill beat the vacuum (see README)
OWL_SPEED = 2.2       # slower than mice (3.0): fleeing straight away opens
                    # distance, and the small catch radius means the owl must
                    # truly intercept — blind drifters still die (~14/32), but
                    # skilled flee+wall play is literally untouchable (0/32)
OWL_TURN = 0.045       # ...but turns wide, so early cutaways work
MOUSE_SPEED = 3.0
TURN_RATE = 0.06      # mouse steering agility
ARENA_MARGIN = 7       # mice bounce inside this inset (no edge clipping)
MAX_TRAIL = 20

# Brain (inputs grow per level; hidden/output fixed)
HIDDEN_NODES = 9
OUTPUT_NODES = 1  # single steering value in [-1, 1]

# GA (selection for SURVIVAL now: higher fitness = lived longer)
POPULATION_SIZE = 50
EVAL_EPISODES = 5  # was 3: with seeds now fixed per level (not per gen, see
                    # main.py), fitness is only ever proven against these
                    # exact layouts -- more of them means "beating the bar"
                    # asks for a more general reflex, not a memorized answer
                    # to 3 specific spawns. The speed fixes in model.py/
                    # mice.py/environment.py paid for roughly this much
                    # headroom without training taking longer than before.
MUTATION_RATE = 0.05
MUTATION_STRENGTH = 0.07  # small steps: keeps children near parents (heritable)
ELITE_FRACTION = 0.2
EPISODE_LENGTH = 1800  # 30s at 60fps

# Fitness shaping: deaths dominate, survival time breaks ties. Must exceed
# EPISODE_LENGTH -- otherwise a mutant that saves ONE FEWER mouse but
# survives much longer on average can out-fitness a mutant that genuinely
# saves more, since 1000 < 1800 lets the time term flip the ordering. At
# 2000 > EPISODE_LENGTH, one extra survivor mathematically always wins
# regardless of the time term, so argmax(fit) always means "most mice
# saved, longest survival among those" -- never the reverse.
SURVIVOR_WEIGHT = 2000  # one saved mouse outweighs any time gain (max EPISODE_LENGTH)
RESPAWN_MIN_DIST = 100  # spawn only (caught mice stay dead: countdown)

MAX_CHECKPOINTS = 30

# Level exits: never cut a rising curve. An earned gate still trains until the
# best flatlines (no gain >= LEVEL_IMPROVE_EPS for LEVEL_STABLE_GENS gens).
# Fixed layouts + elitism make best non-decreasing within a level (the same
# elite, scored on the same episodes, can't score worse), so a flat clock
# means a genuine plateau, not noise. This is the fix for levels exiting the
# instant a threshold was crossed by luck, before the school actually
# mastered that level's sense. A cleared gate still trains until the best
# flatlines, so rising curves are never cut short.
LEVEL_STABLE_GENS = 5    # flat gens (after min_gens) before an earned gate opens
LEVEL_IMPROVE_EPS = 1.0  # gains smaller than this don't reset the plateau clock

# Survivor gates: each level must save one whole mouse more than the
# previous level's clearing average (fractional means don't count) -- time
# alone can never open a door. L1 is the baseline and clears on plateau;
# the finale caps at the full school of 32. An unreachable gate wastes
# nights forever, so autobar below stays as the loud last resort.
AUTOBAR_AFTER = 30    # flat gens at one level before its requirement lowers
AUTOBAR_WINDOW = 10   # recent champion survivor-counts forming the evidence

# Stagnation shake: the plateau gate above answers "has this level's fitness
# stopped moving" -- it can't tell a genuine ceiling from a GA that's simply
# converged onto a mediocre local optimum (low population diversity, same
# handful of genotypes recombined every generation). If a plateau runs on
# for SHAKE_AFTER gens without crossing the bar, temporarily widen the
# search instead of just waiting: elites are still carried over unchanged
# (nothing already proven is ever lost), but the rest of the population gets
# stronger mutation plus a slice of brand-new random immigrants for
# SHAKE_DURATION generations, then settles back to normal. If it's STILL
# stuck after that, autobar above remains the ultimate backstop.
SHAKE_AFTER = 15            # extra stuck gens before a shake fires
SHAKE_MUTATION_MULT = 4.0   # temporary multiplier on rate + strength
SHAKE_IMMIGRANTS = 0.2      # fraction of the population replaced with fresh random brains
SHAKE_DURATION = 5          # generations the boost stays in effect

# --- Level progression: each level grows the prey's senses. Clearing one
# level demands one more saved mouse than the last clearing average (see the
# [wall] line on the console) -- the ladder measures the school against
# itself, in mice, never in abstract fitness. ---
LEVELS = [
    {"name": "blind",     "inputs": ["bias"], "min_gens": 10},
    {"name": "proximity", "inputs": ["bias", "dist"], "min_gens": 10},
    {"name": "direction", "inputs": ["bias", "dist", "dir x", "dir y"],
     "min_gens": 10},
    {"name": "intent",    "inputs": ["bias", "dist", "dir x", "dir y", "closing"],
     "min_gens": 10},
    {"name": "walls",     "inputs": ["bias", "dist", "dir x", "dir y", "closing",
                                     "wall \u2191", "wall \u2193", "wall \u2192", "wall \u2190"],
     "min_gens": 10},
    {"name": "full sense", "inputs": ["bias", "dist", "dir x", "dir y", "closing",
                                      "wall \u2191", "wall \u2193", "wall \u2192", "wall \u2190",
                                      "aim x", "aim y"], "min_gens": 10},
]