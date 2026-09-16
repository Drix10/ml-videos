# Shark Neuroevolution

Shark learns to hunt fish via genetic algorithm. Top panel = live brain (pink = strong weight), bottom = arena.

## Run

    pip install -r requirements.txt
    python main.py

"Evaluating N/M" = scoring the population (window stays responsive). Then the best brain plays live. Press SPACE to skip a replay.

## What you see

- **Gray circles (left):** sensory inputs (4 nearest fish dx,dy). Bright = active.
- **Gray circles (middle):** hidden layer (ReLU).
- **Pink circle (right):** steering output (ax, ay).
- **Lines:** weights. Bright pink + thick = strong, faint gray = ~0.
- **HUD:** `Level` = generation, right-aligned `FISH LEFT` with pink fish icon, live `fitness` of the replay (not the training score), rotating caption. Window title shows gen + best.

## Experiments (see `config.py`)

- Bigger brain: raise `HIDDEN_NODES` (12 → 24).
- Explore vs exploit: raise `MUTATION_STRENGTH` (0.1 → 0.3) for chaos, lower for fine-tuning.
- Harder task: raise `NUM_FISH`; faster evolution: raise `POPULATION_SIZE`.
- Checkpoints: `checkpoints/best_gen_{N}.pt` (last 30 kept), curve: `logs/fitness.csv` (appended, never wiped).

## How it works

`main.py` → `simulate()` (`environment.py`, headless) scores each brain →
`next_generation()` (`evolution.py`: top 20% elites + tournament/crossover/mutate) →
best brain replayed live with `visualizer.draw()`.
