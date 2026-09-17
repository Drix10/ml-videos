# Night Hunt

An owl hunts 32 mice in the dark. The mice share one tiny brain that
**evolves to keep them alive**. It starts blind and grows new senses level
by level — until the school slips past the owl untouched.

Top half: the brain, thinking live. Bottom half: the hunt.
Exact 540×960 (9:16) — record with no scaling.

## What it is trying to do

Show evolution the way it actually feels: nobody programs the mice to flee.
Random brains try, most die, the survivors breed, and escape behavior appears
on its own. Each level gifts one new sense and you watch the brain rewire
around it.

## How it works (simple version)

- Every mouse in a school runs the **same brain**: senses in, turn direction out.
- One episode = 1800 frames (30s) of owl vs school. Score = mean frames lived.
- The owl (2.2) is slower than mice (3.0) and only a true hit counts
  (catch radius 7), so fleeing straight away genuinely escapes — while blind
  drifters still get vacuumed (~14/32). That gap is what each new sense
  climbs, level by level: **success = fewer mice dying each level**.
  Skilled flee-plus-wall play is literally untouchable (0/32).
- Each brain plays 5 identical layouts per generation (fair comparison).
- Best 20% survive unchanged. The rest are bred from winners + small mutations.
- A level clears after 10+ generations with the bar beaten AND the best
  flatlined (no +1 gain for 5 gens) — rising curves are never cut short.
  Each new wall is built from the clearing fitness: next multiple of 25
  above it, +25 (`[wall]` on console). The ladder measures the school
  against itself; brains grow the next sense, keeping everything learned.

## Two ways to run

- `python main.py` — trains all levels headless (no window, console only)
  until the L6 finale wall (set from L5's clearing fitness) clears.
  Leave it running; close anytime.
- `python main.py showcase` — the finished video: each level's all-time best
  replay with level-up cards in between. `SPACE` jumps ahead.
  `python main.py showcase 2 5` replays just those levels.
- Background: `MAX_GENS=200 nohup python main.py > train.log 2>&1 &`
- `RESUME=1 python main.py` — picks up level/gen/population from
  `checkpoints/resume.pt`. A killed night loses nothing.
- `[autobar]` — if a level stalls 30 gens, its bar drops to the median of
  recent bests rounded down to a multiple of 25 (announced on console).
  The new bar is always already-beaten fitness, so a flat plateau clears
  instead of grinding. Walls lower themselves; no night is ever wasted
  on an unreachable number.

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
| 4 · intent | is it closing in | 1300 (then auto) |
| 5 · walls | where the edges are | auto: L4 best + notch |
| 6 · full sense | exact owl position | auto: L5 best + notch (finale) |

## Reading the screen

- **Brain panel** is monochrome: gray rings + labels name each sense,
  line width alone hints at connection strength. No color-coding in there.
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
the bar to already-beaten fitness (median of recent bests, snapped down to
a multiple of 25). Falling `lost-mean` in the console means it's learning. Interrupted? `RESUME=1 python main.py` picks up
where it died. Fresh physics change? Delete `checkpoints/` and
`logs/fitness.csv` and restart clean so old walls don't linger.
