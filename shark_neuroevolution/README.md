# Shark Neuroevolution — Progressive Senses

A shark learns to hunt fish with a small neural network trained by a **genetic
algorithm** (no gradients). The viral mechanic: the brain **starts blind**
(1 input: `bias`) and **grows new senses level by level** — proximity,
direction, intent, walls, full aim. The top panel shows the brain thinking
**live**, new pink nodes appearing as senses unlock.

Portrait 540×960 window — ready to record for Reels/Shorts with no cropping.

## Features

- Progressive brain: 1 → 11 inputs across 6 levels, old weights preserved on level-up.
- MLP (PyTorch): variable inputs → 9 ReLU hidden → single tanh steering output.
- Genetic algorithm per level: elites + tournament + crossover + Gaussian mutation.
- Live visualizer: labeled sensor nodes, weight-colored connections, glowing hot nodes.
- Arena: angle-steered shark with trail, fleeing fish with tails, eat-burst rings.
- HUD: `Level X / 6 · name`, `ree[g]orithm` tag, live fish countdown, caption bar.
- Logging: per-level checkpoints + append-only fitness CSV (`level` column).

## Requirements

- Python 3.10+ (tested on 3.13)
- `pip install -r requirements.txt` → `pygame`, `torch`, `numpy`
- No GPU needed.

## Run

```bash
cd shark_neuroevolution
pip install -r requirements.txt
python main.py
```

What happens:

1. **Evaluating N/M** — all 50 brains play headless for the current level.
2. **Replay** — the level's best brain plays live with its brain diagram above.
3. When best fitness beats the level threshold → **LEVEL UP**: every brain
   grows the new sensor inputs (old knowledge kept, new weights start quiet).
4. Ends at Level 6 `full sense` — evolution continues there indefinitely.

### Controls

| Key | Action |
|---|---|
| `SPACE` / `ESC` | Skip the current replay, jump to next generation |
| Window ✕ | Quit (log flushed, safe anytime) |

### Files created at runtime

| Path | Contents |
|---|---|
| `checkpoints/best_L{L}_gen{G}.pt` | Best brain per level+generation (last 30 kept) |
| `logs/fitness.csv` | `level,generation,best,mean,worst` — appended across runs |

To replay a saved brain: `Brain.load(path, input_size)` with the level's
input count (1, 2, 4, 5, 9, 11), then step an `Arena(brain, level_idx)`.

## The 6 levels (`config.py`)

| Level | Name | Senses | To pass |
|---|---|---|---|
| 1 | blind | `bias` | 150 | above blind-luck ceiling (~112) |
| 2 | proximity | + `dist` | 200 | must beat evolved blind |
| 3 | direction | + `dir x`, `dir y` | 280 | randoms can't fake aiming anymore |
| 4 | intent | + `closing` | 330 | chase, don't bump |
| 5 | walls | + 4 wall sensors | 380 (~37 fish) | near-perfect runs |
| 6 | full sense | + `aim x`, `aim y` | finale, evolves forever | perfect-aimer ceiling: 409 |

## What you see

### Brain panel (top)

- **Labeled input nodes (left):** only the current level's senses exist —
  watch `dist`, `dir x`… appear in pink on level-up. Bright ring = live signal.
- **Hidden (middle):** 9 ReLU neurons, faint gray.
- **Output (right):** single pink steering node (`[-1, 1]` → turn rate).
- **Lines:** `|weight|` → pink thickness; near-zero → faint gray.

### Arena (bottom)

- White triangle shark (pink glow ring, fading blue trail), white fish with
  tail fins (wander + flee), pink expanding ring on each catch.
- HUD: `ree[g]orithm` top-left, `N / 40 FISH LEFT` top-right.
- Bottom caption bar shows the level's caption; `SPACE skip replay` hint below.

## How it works

```
main.py
 ├─ simulate(brain, level)   environment.py  # headless episode → fitness
 ├─ next_generation()        evolution.py    # elites + tournament + xover + mutate
 ├─ Brain.grow(new_size)     model.py        # level-up: copy old cols, quiet new cols
 ├─ Arena.step()             environment.py  # live replay (SAME physics fn)
 └─ visualizer.*             visualizer.py   # draw_network / draw_arena / draw_caption
agent.py: Agent (level-dependent senses, angle steering, wall bounce)
fish.py:  Fish (wander + proximity flee, distanced spawn, dead when eaten)
config.py: layout, physics, GA, LEVELS, LEVEL_THRESHOLDS
```

**Episode:** shark senses → steering turns it (`angle += out × 0.15`) →
constant-speed swim → fish flee → contact `< 15 + 5` eats (`+10`), dead fish
stay dead, countdown runs. All eaten → `+100` time bonus. Every frame costs
`IDLE_PENALTY`, every wall bounce costs `WALL_PENALTY` (same in training and
replay — earlier versions only penalized the replay).

**Fitness** = `10 × eaten + 100 (if cleared) − 0.005 × frames − 0.5 × bounces`.

## Experiments (`config.py`)

| Change | Effect | Episode idea |
|---|---|---|
| `POPULATION_SIZE` 50 → 100 | Faster learning, slower eval | "100 sharks" |
| `EPISODE_LENGTH` 1800 → 900 | 15 s gens, faster series | "Speedrun evolution" |
| `LEVEL_THRESHOLDS` lower | Rush to full sense | "Zero to predator" |
| `MUTATION_STRENGTH` 0.1 → 0.3 | Chaos, escapes local optima | "Too much mutation" |
| `NUM_FISH` 40 → 60 | Denser, easier early levels | "Overpopulated arena" |
| `FISH_SPEED` 3.0 → 3.5 | Prey arms race | "Faster fish" |
| `HIDDEN_NODES` 9 → 16 | Bigger brain | "Does brain size matter?" |

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: config` | Run from inside `shark_neuroevolution/` |
| Stuck on one level 30+ gens | Lower that level's threshold, or raise `MUTATION_STRENGTH` to 0.2 |
| Levels fly by too fast | Raise thresholds (early ones especially) |
| Shark spins forever | Local optimum — bump `MUTATION_STRENGTH` for a few gens |
| Old checkpoint won't load | Input size changed between levels — checkpoints are per-level dims; use matching `input_size` in `Brain.load` |

## Recording guide (video series)

**Setup:** 540×960 portrait — native for Reels/Shorts/TikTok, record at
1080×1920 (2× scale, crisp pink lines). 60 fps. No game audio exists — narrate
in post.

**What to capture:**

1. **Ep 1 — blind:** gen 0 replay (circles, gray fuzz) → the LEVEL UP moment
   (`dist` node appears). The before/after IS the video.
2. **Eps 2–5:** one level each. Record the level-up replay (new nodes glowing)
   + the generation where `eaten` first jumps (console `best` spikes).
3. **Ep 6 — full sense:** the "ultimate predator" montage: 2–3 best replays
   back-to-back + the `logs/fitness.csv` curve for the thumbnail.

**When to hit record:**

- ✅ Gen 0 of every level (fresh senses, chaotic behavior).
- ✅ The LEVEL UP console moment + following replay.
- ✅ Any replay with catches + pink highways strengthening.
- ⏩ SKIP with SPACE: plateau generations (console `best` flat 3+ gens) —
  show the CSV curve instead of boring viewers.
- 🏁 Finale: full-sense best brain, 3 replays stitched.

**Reading the console on camera:** `best` climbing + `mean` following = learning.
`best` high but `mean` flat = one lucky shark, keep evolving. Both flat 20+
gens = stuck — raise mutation live and restart the level.
