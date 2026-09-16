# Night Hunt — prey-side neuroevolution

A fixed **owl** hunts 32 **mice**, and the mice evolve. Each mouse shares one
small neural network (the species brain), trained by a **genetic algorithm**
on survival time. The brain starts blind (`bias` only) and grows new senses
level by level. Top panel = the prey's live brain; bottom = the night arena.

Portrait 540×960 window — record for Reels/Shorts with no cropping.

## Features

- Progressive prey brain: 1 → 11 inputs across 6 levels, old weights kept.
- MLP (PyTorch): variable inputs → 9 hidden → single steering output.
- Whole school through **one batched forward pass** per step (32× cheaper).
- Fixed owl hunter: turn-limited pursuit, wide arcs — early cutaways work.
- Gold-on-midnight UI: ring graph, owl silhouette, oriented mice, catch burst.
- Logging: per-level checkpoints + append-only fitness CSV with catch stats.

## Requirements

- Python 3.10+ (tested on 3.13)
- `pip install -r requirements.txt` → `pygame`, `torch`, `numpy`
- No GPU needed.

## Run

```bash
cd night_hunt
pip install -r requirements.txt
python main.py
```

1. **Evaluating N/M** — 50 brains, 3 episodes each (averaged: luck spikes die).
2. **Replay** — the best school plays live under its brain diagram.
3. Beat the survival bar after 10+ gens → **LEVEL UP**: brains grow new senses.
4. Level 6 evolves forever. `SPACE` skips replays. `MAX_GENS=N SKIP_REPLAY=1`
   env vars give capped headless monitor runs.

### Files created at runtime

| Path | Contents |
|---|---|
| `checkpoints/best_L{L}_gen{G}.pt` | Best brain per level+generation (last 30 kept) |
| `logs/fitness.csv` | `level,generation,best,mean,worst,catch_best,catch_mean` |

Replay a saved brain: `Brain.load(path, input_size)` (1/2/4/5/9/11 per level),
then step an `Arena(brain, level_idx)`.

## The 6 levels (`config.py`)

| Level | Name | Senses | To pass (mean survived frames) |
|---|---|---|---|
| 1 | blind | `bias` | 1200 |
| 2 | proximity | + `dist` | 1350 |
| 3 | direction | + `dir x`, `dir y` | 1500 |
| 4 | intent | + `closing` | 1600 |
| 5 | walls | + 4 wall sensors | 1750 (near-untouchable) |
| 6 | full sense | + `aim x`, `aim y` | finale, evolves forever |

Calibrated: random schools ~1000, heuristic escaper ~1320–1590, max 1900.

## What you see

### Brain panel (top)

- **Labeled sensor rings:** only the level's senses exist; the newest glow gold.
- **Hidden rings + gold output ring.** Thin gray fan, thick gold highways.
- **Two-tone level line** under the graph.

### Arena (bottom)

- White owl (gold eye, presence ring, flight trail) vs moon-white mice
  (ears, tails, oriented to heading). Gold burst on each catch.
- Counter: gold `N / 32 MICE LEFT`. Bottom caption bar per level.

## How it works

```
main.py
 ├─ simulate(brain, level)   environment.py  # headless episode → survival fit
 ├─ next_generation()        evolution.py    # elites + tournament + xover + mutate
 ├─ Brain.grow(new_size)     model.py        # level-up keeps learned weights
 ├─ Brain.act_batch(obs)     model.py        # whole school, one forward pass
 ├─ Arena.step()             environment.py  # live replay (SAME physics)
 └─ visualizer.*             visualizer.py   # rings graph / arena / caption
owl.py:   Owl (fixed PD pursuit hunter — the exam, never evolves)
mice.py:  Mouse (brain-steered prey, survival clock, distanced spawn)
config.py: layout, colors, physics, GA, LEVELS + thresholds
```

**Episode:** owl steers toward nearest mouse (turn-limited) → every live mouse
senses → one batched brain pass → all steer → contact < `CATCH_RADIUS` kills.
Fitness = mean survived frames (+100 if nobody caught).

## Experiments (`config.py`)

| Change | Effect |
|---|---|
| `OWL_TURN` 0.045 → 0.08 | Tighter hunter, escape must be perfect |
| `OWL_SPEED` 3.2 → 3.5 | Faster threat, shorter schools |
| `NUM_MICE` 32 → 48 | Richer footage, easier early survival |
| `EVAL_EPISODES` 3 → 1 | Faster gens, luck spikes return (not advised) |
| `LEVEL_THRESHOLDS` per level | Pace the series |

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: config` | Run from inside `night_hunt/` |
| Stuck on a level 30+ gens | Lower its `threshold`; check `catch_mean` is climbing |
| `catch_mean` flat, `best` spiky | Raise `EVAL_EPISODES` (variance too high) |
| Old checkpoint won't load | Pass the level's `input_size` to `Brain.load` |
| `fitness.csv locked` message | Another live run holds it; session file used instead |

## Recording guide

540×960 portrait, 1080×1920 @60fps capture, narrate in post.

- ✅ Gen 0 of every level (new gold senses, naive school).
- ✅ LEVEL UP moment + following replay.
- ✅ First replay where the school visibly parts around the owl.
- ⏩ SKIP plateaus (`best` flat 3+ gens) — show the CSV curve instead.
- 🏁 Finale: full-sense school, near-zero catches, 2–3 replays stitched.
- Read the console: `catch_mean` climbing = learning; `best` spiky + flat mean
  = luck; both flat 20+ gens = lower the bar live.
