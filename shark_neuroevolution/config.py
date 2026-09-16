# Portrait UI (phone/Reel-ready) + level progression
WIDTH, HEIGHT = 540, 960
FPS = 60

# Layout
NETWORK_RECT = (20, 44, 500, 420)   # brain diagram panel
GAME_RECT = (20, 484, 500, 380)     # arena panel (arena coords are 500x380)
ARENA_W, ARENA_H = 500, 380
CAPTION_Y = 915

# Colors
BG_COLOR = (18, 18, 18)
ARENA_COLOR = (24, 24, 24)
SHARK_COLOR = (255, 255, 255)
FISH_COLOR = (220, 220, 220)
PINK = (255, 0, 128)      # active / strong
DIM_GRAY = (70, 70, 70)   # weak lines: visible on BG, still clearly "untrained"
TEXT_WHITE = (240, 240, 240)
TEXT_GRAY = (120, 120, 120)

# Game
NUM_FISH = 32
SHARK_RADIUS = 15
FISH_RADIUS = 5
CATCH_RADIUS = 15  # edible distance; shrink to punish accidental bumps
SHARK_SPEED = 3.5     # faster than fish: hunts must chase, not just bump
FISH_SPEED = 2.8
TURN_RATE = 0.05        # slow deliberate turns (was 0.15 spin)
FLEE_RADIUS = 200       # fish sense danger further out
FLEE_STRENGTH = 4.0
MAX_TRAIL = 20

# Brain (inputs grow per level; hidden/output fixed)
HIDDEN_NODES = 9
OUTPUT_NODES = 1  # single steering value in [-1, 1]

# GA
POPULATION_SIZE = 50
MUTATION_RATE = 0.05
MUTATION_STRENGTH = 0.1
ELITE_FRACTION = 0.2
EPISODE_LENGTH = 1800  # 30s at 60fps

# Fitness shaping
EAT_REWARD = 10.0
TIME_BONUS = 100.0   # all fish eaten before timeout
IDLE_PENALTY = 0.005
WALL_PENALTY = 0.5
RESPAWN_MIN_DIST = 100  # spawn only (fish die when eaten, no respawn)

MAX_CHECKPOINTS = 30

# --- Level progression: the viral mechanic. Each level adds senses. ---
LEVELS = [
    {"name": "blind",     "inputs": ["bias"],
     "caption": "senses, no training, so",
     "description": "Only knows it exists. Flails randomly.",
     "threshold": 150, "min_gens": 10},
    {"name": "proximity", "inputs": ["bias", "dist"],
     "caption": "Give every fish one",
     "description": "Knows how far the nearest fish is.",
     "threshold": 200, "min_gens": 10},
    {"name": "direction", "inputs": ["bias", "dist", "dir x", "dir y"],
     "caption": "coming from, and they",
     "description": "Knows the exact direction to the nearest fish.",
     "threshold": 280, "min_gens": 10},
    {"name": "intent",    "inputs": ["bias", "dist", "dir x", "dir y", "closing"],
     "caption": "Feed it the predator's",
     "description": "Knows if it is closing in or drifting away.",
     "threshold": 330, "min_gens": 10},
    {"name": "walls",     "inputs": ["bias", "dist", "dir x", "dir y", "closing",
                                     "wall \u2191", "wall \u2193", "wall \u2192", "wall \u2190"],
     "caption": "Add wall sensors so",
     "description": "Knows where the walls are to avoid crashing.",
     "threshold": 380, "min_gens": 10},
    {"name": "full sense", "inputs": ["bias", "dist", "dir x", "dir y", "closing",
                                      "wall \u2191", "wall \u2193", "wall \u2192", "wall \u2190",
                                      "aim x", "aim y"],
     "caption": "Full senses now, and",
     "description": "Perfect aim. The ultimate predator.",
     "threshold": float("inf"), "min_gens": float("inf")},  # finale: evolves forever
]
