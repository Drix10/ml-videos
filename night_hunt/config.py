# Night hunt: fixed owl vs evolving mice. Portrait UI (phone/Reel-ready).
WIDTH, HEIGHT = 540, 860  # logical units; x UI_SCALE on screen (1080x1920)
FPS = 60
UI_SCALE = 1  # 1 = 540x860 watch window; 2 = crisp 1080x1920 recording

# Layout
NETWORK_RECT = (20, 44, 500, 340)   # brain diagram panel
GAME_RECT = (20, 400, 500, 380)     # arena panel (arena coords are 500x380)
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
CATCH_RADIUS = 12
OWL_SPEED = 3.2       # fixed threat: faster than any mouse
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
EVAL_EPISODES = 3  # fitness = mean over episodes: one lucky run can't spike
MUTATION_RATE = 0.05
MUTATION_STRENGTH = 0.1
ELITE_FRACTION = 0.2
EPISODE_LENGTH = 1800  # 30s at 60fps

# Fitness shaping
SURVIVE_BONUS = 100.0  # nobody caught before timeout
RESPAWN_MIN_DIST = 100  # spawn only (caught mice stay dead: countdown)

MAX_CHECKPOINTS = 30
SHOW_BEST_ONLY = True  # video mode: train headless, replay only the level-best

# --- Level progression: each level grows the prey's senses. ---
LEVELS = [
    {"name": "blind",     "inputs": ["bias"],
     "caption": "no senses, no fear, so",
     "description": "Knows nothing. Drifts.",
     "threshold": 1100, "min_gens": 10},
    {"name": "proximity", "inputs": ["bias", "dist"],
     "caption": "feel how near it",
     "description": "Bolts when it nears — but which way?",
     "threshold": 1150, "min_gens": 10},
    {"name": "direction", "inputs": ["bias", "dist", "dir x", "dir y"],
     "caption": "sense where from, and",
     "description": "Finally flees the right way.",
     "threshold": 1400, "min_gens": 10},
    {"name": "intent",    "inputs": ["bias", "dist", "dir x", "dir y", "closing"],
     "caption": "read the lunge before",
     "description": "Cuts away early; the owl overshoots.",
     "threshold": 1500, "min_gens": 10},
    {"name": "walls",     "inputs": ["bias", "dist", "dir x", "dir y", "closing",
                                     "wall \u2191", "wall \u2193", "wall \u2192", "wall \u2190"],
     "caption": "learn the edges so",
     "description": "Stops cornering itself.",
     "threshold": 1650, "min_gens": 10},
    {"name": "full sense", "inputs": ["bias", "dist", "dir x", "dir y", "closing",
                                      "wall \u2191", "wall \u2193", "wall \u2192", "wall \u2190",
                                      "aim x", "aim y"],
     "caption": "untouchable now, and",
     "description": "A brain from nothing. Still here.",
     "threshold": float("inf"), "min_gens": float("inf")},  # finale: evolves forever
]
