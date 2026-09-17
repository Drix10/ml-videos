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

# Fitness shaping
SURVIVE_BONUS = 100.0  # nobody caught before timeout
RESPAWN_MIN_DIST = 100  # spawn only (caught mice stay dead: countdown)

MAX_CHECKPOINTS = 30
FINALE_TARGET = 1800  # fallback only: normally overridden by the dynamic
                       # wall below the moment training reaches the finale

# Level exits: never cut a rising curve. A beaten bar still trains until the
# best flatlines (no gain >= LEVEL_IMPROVE_EPS for LEVEL_STABLE_GENS gens).
# Fixed layouts + elitism make best non-decreasing within a level (the same
# elite, scored on the same episodes, can't score worse), so a flat clock
# means a genuine plateau, not noise. This is the fix for levels exiting the
# instant a threshold was crossed by luck, before the school actually
# mastered that level's sense.
LEVEL_STABLE_GENS = 5    # flat gens (after min_gens) before a beaten bar opens the door
LEVEL_IMPROVE_EPS = 1.0  # gains smaller than this don't reset the plateau clock

# Dynamic walls: L2..L6's bar is built from the PREVIOUS level's clearing
# fitness (next clean multiple of AUTOBAR_ROUND above it, + NEXT_BAR_MARGIN)
# instead of a fixed guess pulled from nowhere. Only L1's threshold below is
# a fixed number now; it only has to be low enough that "blind" mice, who
# have nothing to work with but bias, can clear it at all.
AUTOBAR_AFTER = 30    # stuck gens at one level before its wall lowers (backstop)
AUTOBAR_WINDOW = 10   # recent gen-bests forming the evidence
AUTOBAR_ROUND = 25    # bars snap to multiples of this -- no odd numbers
NEXT_BAR_MARGIN = 25  # new wall = clearing fitness snapped up + this notch

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

# --- Level progression: each level grows the prey's senses. Only L1's
# threshold below is used as-is; every later level's real bar is set
# dynamically at level-up time (see NEXT_BAR_MARGIN above and the [wall]
# line on the console) from what the population just proved it could do. ---
LEVELS = [
    {"name": "blind",     "inputs": ["bias"],
     "threshold": 1100, "min_gens": 10},
    {"name": "proximity", "inputs": ["bias", "dist"],
     "threshold": 1150, "min_gens": 10},
    {"name": "direction", "inputs": ["bias", "dist", "dir x", "dir y"],
     "threshold": 1230, "min_gens": 10},
    {"name": "intent",    "inputs": ["bias", "dist", "dir x", "dir y", "closing"],
     "threshold": 1300, "min_gens": 10},
    {"name": "walls",     "inputs": ["bias", "dist", "dir x", "dir y", "closing",
                                     "wall \u2191", "wall \u2193", "wall \u2192", "wall \u2190"],
     "threshold": 1550, "min_gens": 10},
    {"name": "full sense", "inputs": ["bias", "dist", "dir x", "dir y", "closing",
                                      "wall \u2191", "wall \u2193", "wall \u2192", "wall \u2190",
                                      "aim x", "aim y"],
     "threshold": float("inf"), "min_gens": 10},  # finale bar is set dynamically
]