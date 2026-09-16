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
- One episode = 1800 frames (30s) of owl vs school. Score = mean frames lived.
- Each brain plays 3 identical layouts per generation (fair comparison).
- Best 20% survive unchanged. The rest are bred from winners + small mutations.
- Beat the level's bar after 10+ generations → brains grow the next sense,
  keeping everything already learned.

## Two ways to run

- `python main.py` — trains all levels headless (no window, console only)
  until the L6 finale clears 1800. Leave it running; close anytime.
- `python main.py showcase` — the finished video: each level's all-time best
  replay with level-up cards in between. `SPACE` jumps ahead.
  `python main.py showcase 2 5` replays just those levels.
- Background: `MAX_GENS=200 nohup python main.py > train.log 2>&1 &`
- `RESUME=1 python main.py` — picks up level/gen/population from
  `checkpoints/resume.pt`. A killed night loses nothing.
- `[autobar]` — if a level stalls 30 gens, its bar drops to the median of
  recent bests + 10 (announced on console). Walls lower themselves; no night
  is ever wasted on an unreachable number.

## Run it

```bash
cd night_hunt
pip install -r requirements.txt
python main.py          # headless training, console only
python main.py showcase # the video, once training finishes
```

Training never opens a window. Replays run at 60 FPS; training is
frame-counted, so only replay pace changes. Run one trainer per folder
(rows interleave otherwise).

## The 6 levels

| Level | New sense | To pass |
|---|---|---|
| 1 · blind | nothing (just `bias`) | 1100 |
| 2 · proximity | how close the owl is | 1150 |
| 3 · direction | where it's coming from | 1230 |
| 4 · intent | is it closing in | 1300 |
| 5 · walls | where the edges are | 1550 |
| 6 · full sense | exact owl position | 1800 (finale) |

## Reading the screen

- **Gold rings/labels** = this level's new senses (older ones glow gold only
  while firing). Gray = quiet. Thick gold lines = strong connections.
- **Counter** top-right = mice still alive. New senses slide into the graph;
  catches burst gold particles; the arena wears a soft vignette.

## Files

`config.py` (all settings) · `model.py` (brain) · `mice.py` (prey) ·
`owl.py` (fixed hunter) · `environment.py` (episodes) ·
`evolution.py` (breeding) · `visualizer.py` (drawing) · `main.py` (run it)

Env knobs: `MAX_GENS` (cap a session) · `RESUME=1` (pick up where it died) ·
`SHOWCASE=1` (same as `showcase`) · `LOG_DIR`/`CKPT_DIR` (isolated test runs).

## If it gets stuck

It unsticks itself: 30 stalled generations triggers `[autobar]`, which drops
the bar to the median of recent bests + 10. Falling `lost-mean` in the
console means it's learning. Interrupted? `RESUME=1 python main.py` picks up
where it died.
