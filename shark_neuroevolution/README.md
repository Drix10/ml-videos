# Shark Neuroevolution

A shark learns to hunt fish with its own small neural network, trained by a
**genetic algorithm** (neuroevolution — no gradients). The top panel shows the
brain thinking **live**: every weight and activation, updated each frame.
The bottom panel is the arena.

![layout](https://dummyimage.com/800x200/08080e/ff46b4&text=LIVE+BRAIN:+pink+%3D+strong+weight)
![arena](https://dummyimage.com/800x200/08080e/ffffff&text=arena:+white+shark+hunts+white+fish)

## Features

- MLP brain (PyTorch): configurable inputs → hidden ReLU → tanh steering.
- Genetic algorithm: elites + tournament selection + crossover + Gaussian mutation.
- Live brain visualizer: weight-colored connections, glowing active nodes, ax/ay steering bars.
- Game HUD: generation level, fish counter, live fitness, rotating explainer captions.
- Juice: shark motion trail, pink eat-burst ring, eval progress bar.
- Logging: per-generation checkpoints + append-only fitness CSV.

## Requirements

- Python 3.10+ (tested on 3.13)
- `pip install -r requirements.txt` → installs `pygame`, `torch`, `numpy`
- No GPU needed. CPU-only torch is fine.

## Run

```bash
cd shark_neuroevolution
pip install -r requirements.txt
python main.py
```

What happens:

1. **Evaluating N/M** — all 50 brains play headless (fast, no rendering).
   The window stays responsive with a progress bar.
2. **Replay** — the best brain of the generation plays live in the arena
   with its brain diagram above it (~30 s or until all fish are eaten).
3. Repeat. Watch the shark go from flailing to hunting over 20–50 generations.

### Controls

| Key | Action |
|---|---|
| `SPACE` / `ESC` | Skip the current replay, jump to next generation |
| Window ✕ | Quit (fitness log is flushed, safe anytime) |

### Files & folders created at runtime

| Path | Contents |
|---|---|
| `checkpoints/best_gen_{N}.pt` | Best brain weights per generation (last 30 kept) |
| `logs/fitness.csv` | `generation,best,mean,worst` — appended across runs |

## What you see

### Brain panel (top)

- **SENSORS (left, gray):** 8 inputs = 4 nearest fish as `(dx, dy)` in `[-1, 1]`.
  Bright = large signal (fish far off-axis).
- **HIDDEN (middle, gray):** 12 ReLU neurons. Glow ring = highly active.
- **STEERING (right, pink):** 2 outputs `(ax, ay)` in `[-1, 1]`, with live value
  bars underneath growing left/right from center.
- **Lines:** connection weights. Bright pink + thick = strong (`|w| ≳ 0.5`),
  faint gray + thin = near zero. Header shows the live **strong-link count** —
  it starts near-random and a few pink highways emerge as it learns.

### Arena (bottom)

- White triangle = shark (rotates to velocity), fading blue trail = recent path.
- White ellipses = fish (wander + flee when the shark is close).
- Pink expanding ring = a catch.
- HUD top-left: `Level {gen}` (= generation) + live `fitness`/`eaten`.
  Top-right: pink fish icon + `N / TOTAL FISH LEFT`.
  Bottom-center: rotating caption. Window title: `gen + best training score`.

## How it works

```
main.py
 ├─ simulate(brain)      environment.py   # headless episode → fitness
 ├─ next_generation()    evolution.py     # elites + tournament + crossover + mutate
 ├─ Arena.step()         environment.py   # live replay episode (same physics)
 └─ visualizer.draw()    visualizer.py    # brain diagram from live weights
model.py: Brain (fc1→ReLU→fc2→tanh), mutate(), crossover(), save/load
agent.py: Shark (sense 4 nearest fish, velocity-clamped steering, wall detect)
fish.py:  Fish (wander + proximity flee, distanced respawn)
config.py: every tunable in one place
```

**Episode loop** (`simulate` / `Arena.step`, identical physics):

1. Shark senses → brain outputs `(ax, ay)` → velocity integrates with
   friction (`v = v*0.9 + a*1.5`, clamped to `SHARK_SPEED`).
2. Fish wander + flee, shark moves, wall hits penalized.
3. Shark within `EAT_RADIUS` → `+EAT_REWARD`, fish respawns ≥120px away
   (so sitting still never farms food).
4. Every frame costs `IDLE_PENALTY` (efficiency pressure).

**Fitness** = `10 × eaten − 0.01 × frames − 1 × wall_hits`.
Eating 2+ fish in 30 s beats doing nothing — the first thing evolution discovers.

**Genetic algorithm** (`evolution.py`):

1. Score all `POPULATION_SIZE` brains for `EPISODE_LENGTH` frames.
2. Top `ELITE_FRACTION` (20%) carry over **unchanged**.
3. Rest: pick 2 parents by 3-way tournament, uniform crossover per weight,
   Gaussian mutation (`MUTATION_RATE` of weights += noise × `MUTATION_STRENGTH`).

## Experiments (`config.py`)

| Change | Effect | Episode idea |
|---|---|---|
| `HIDDEN_NODES` 12 → 24 | Bigger brain, slower but smarter | "Does brain size matter?" |
| `POPULATION_SIZE` 50 → 10 / 100 | Slower / faster learning | "10 vs 100 sharks" |
| `MUTATION_STRENGTH` 0.1 → 0.3 | More chaos, less fine-tuning | "Too much mutation" |
| `MUTATION_RATE` 0.05 → 0.2 | More exploration | "Mutation rate sweep" |
| `NUM_FISH` 20 → 40 | Harder task | "Overpopulated arena" |
| `FISH_SPEED` 2.0 → 3.0 | Prey arms race | "Faster fish" |
| `IDLE_PENALTY` 0.01 → 0.05 | Rewards speedsters, punishes drifters | "Lazy sharks die" |

To replay a saved brain without training: load any
`checkpoints/best_gen_{N}.pt` with `Brain.load()` and step an `Arena` —
see `Arena` in `environment.py` (3 lines).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: config` | Run from inside `shark_neuroevolution/` (`cd` there first) |
| Window "not responding" during eval | Normal on slow machines — the progress bar updates between sharks; wait or lower `POPULATION_SIZE`/`EPISODE_LENGTH` |
| Shark spins in circles forever | Local optimum — raise `MUTATION_STRENGTH` to 0.2 for a few gens |
| Nothing learns after 50 gens | Check `logs/fitness.csv`: if `best` flatlines, raise `MUTATION_RATE`; if fish never caught, lower `FISH_SPEED` |
| `torch.load` error on old checkpoint | Checkpoints are state-dicts of the default dims; if you changed `HIDDEN_NODES`, old files won't load — delete `checkpoints/` |

## Recording guide (for the video series)

**Setup:** 800×920 window — capture at 1080p60 with the window top-aligned
(no scaling, so the pink lines stay crisp). Dark room / dark editor behind it
matches the near-black arena. Record system audio off; add narration in post —
the only motion audio is none (no sound in sim).

**What to capture per episode:**

1. **Ep 1 — Baseline (gen 0–5):** record the first replay: shark flailing,
   brain all gray fuzz. Then jump-cut to ~gen 20: first clean catches + pink
   highways appearing. The before/after IS the video.
2. **Ep 2 — Population (10 vs 100):** set `POPULATION_SIZE`, delete `logs/` +
   `checkpoints/`, record gen 0 and gen 10 replays + plot `logs/fitness.csv`
   (`best` column) for the thumbnail graph.
3. **Ep 3+ — one config change per episode** (table above). Always start from
   scratch (fresh `logs/`/`checkpoints/`) so viewers see the full learning arc.

**When to hit record:**

- ✅ First replay of gen 0 (pure chaos — gold).
- ✅ The generation where `eaten` first hits 5+ (check console: `best` jumps).
- ✅ Any replay with 3+ pink strong-links + trail curving into fish.
- ⏩ SKIP with SPACE: mid-training replays where nothing changes (gens 6–15
  often plateau) — don't bore viewers, show the fitness CSV instead.
- 🏁 Finale: best checkpoint replayed 2–3× back-to-back for the "it hunts" montage.

**Reading `logs/fitness.csv` on camera:** `best` climbing + `mean` following it
= learning. `best` high but `mean` flat = one lucky shark, keep evolving.
Both flat for 20+ gens = stuck — bump mutation live in `config.py` and restart.
