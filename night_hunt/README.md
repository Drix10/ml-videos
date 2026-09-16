# Night Hunt

An owl hunts 32 mice in the dark. The mice share one tiny brain that
**evolves to keep them alive**. It starts blind and grows new senses level
by level — until the school slips past the owl untouched.

Top half: the brain, thinking live. Bottom half: the hunt.
Exact 540×960 (9:16) — record with no scaling. Captions sit under the arena.

## What it is trying to do

Show evolution the way it actually feels: nobody programs the mice to flee.
Random brains try, most die, the survivors breed, and escape behavior appears
on its own. Each level gifts one new sense and you watch the brain rewire
around it.

## How it works (simple version)

- Every mouse in a school runs the **same brain**: senses in, turn direction out.
- One episode = 30 seconds of owl vs school. Score = average seconds lived.
- Each brain plays 3 episodes (so one lucky run can't fool us).
- Best 20% survive unchanged. The rest are bred from winners + small mutations.
- Beat the level's score after 10+ generations → brains grow the next sense,
  keeping everything already learned.

## Two ways to run

- `python main.py` — trains all levels headless (no window, console only)
  until the L6 finale clears 1800. Leave it running; close anytime.
- `python main.py showcase` — the finished video: each level's all-time best
  replay with level-up cards in between. `SPACE` jumps ahead.
  `python main.py showcase 2 5` replays just those levels.
- Background: `MAX_GENS=200 nohup python main.py > train.log 2>&1 &`

## Run it

```bash
cd night_hunt
pip install -r requirements.txt
python main.py
```

Watch the best school replay each generation. `SPACE` skips boring ones.
`SHOW_BEST_ONLY` (default on) trains each level headless and replays only
the level-winning school — one showcase replay per stage, no filler.
Level-ups hit a gold flash with the new sense held full-screen (edit point).
Close the window anytime — progress is saved in `logs/` + `checkpoints/`.
Replays run at 60 FPS. Training is frame-counted, so only replay pace changes.

## The 6 levels

| Level | New sense | To pass |
|---|---|---|
| 1 · blind | nothing (just `bias`) | 1100 |
| 2 · proximity | how close the owl is | 1150 |
| 3 · direction | where it's coming from | 1230 |
| 4 · intent | is it closing in | 1300 |
| 5 · walls | where the edges are | 1650 |
| 6 · full sense | exact owl position | evolves forever |

## Reading the screen

- **Gold rings/labels** = this level's new senses (older ones glow gold only
  while firing). Gray = quiet. Thick gold lines = strong connections.
- **Counter** top-right = mice still alive. New senses slide into the graph;
  catches burst gold particles; the arena wears a soft vignette.

## Files

`config.py` (all settings) · `model.py` (brain) · `mice.py` (prey) ·
`owl.py` (fixed hunter) · `environment.py` (episodes) ·
`evolution.py` (breeding) · `visualizer.py` (drawing) · `main.py` (run it)

## If it gets stuck

A level that can't pass in ~25 generations has its bar too high — lower that
level's `threshold` in `config.py` and restart. Falling `lost-mean` in the
console means it's learning.
