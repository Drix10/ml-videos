# All tunables in one place. Tweak these for video episodes.
POPULATION_SIZE = 50
MUTATION_RATE = 0.05
MUTATION_STRENGTH = 0.1
ELITE_FRACTION = 0.2

INPUT_NODES = 8   # 4 nearest fish x (dx, dy), normalized to [-1, 1]
HIDDEN_NODES = 12
OUTPUT_NODES = 2  # (ax, ay) steering, tanh

FPS = 60
EPISODE_LENGTH = 1800  # 30s at 60fps
NUM_FISH = 20

ARENA_W, ARENA_H = 800, 600
PANEL_H = 320  # brain strip; 800x920 window fits 768p+ screens
EAT_RADIUS = 18
RESPAWN_MIN_DIST = 120  # stops fish respawning under the shark (free food hack)
FISH_SPEED = 2.0
SHARK_SPEED = 4.0

EAT_REWARD = 10.0
IDLE_PENALTY = 0.01  # per frame without eating
WALL_PENALTY = 1.0

MAX_CHECKPOINTS = 30  # disk cap: oldest best_gen_*.pt deleted beyond this
TRAIL_LEN = 24  # shark motion trail length (frames)
