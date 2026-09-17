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
- One episode = 1800 frames (30s) of owl vs school. Score = living mice x 2000
  + mean frames lived: **deaths dominate, time breaks ties**. One extra
  survivor (+2000) always beats any survival-time gain (max 1800) —
  mathematically, not just usually.
- The owl (2.2) is slower than mice (3.0) and only a true hit counts
  (catch radius 7), so fleeing straight away genuinely escapes — while blind
  drifters still get vacuumed (~14/32). That gap is what each new sense
  climbs, level by level: **success = fewer mice dying each level**.
  Skilled flee-plus-wall play is literally untouchable (0/32).
- Each brain plays 5 identical layouts per generation (fair comparison).
- Best 20% survive unchanged. The rest are bred from winners + small mutations.
- A level clears after 10+ generations with one more saved mouse earned AND
  the best flatlined (no +1 gain for 5 gens) — rising curves are never cut
  short. Each new requirement is one whole mouse above the clearing average
  (`[wall]` on console). The ladder measures the school against itself, in
  mice; brains grow the next sense, keeping everything learned.

## What's inside the brain

One shared MLP, 9 hidden units, small enough to read in one sitting:

- `fc1` turns the current senses (1 → 3 → 4 → 5 → 9 → 11 across levels)
  into hidden activity; ReLU keeps it nonlinear and cheap.
- `fc2` turns hidden activity into one steering value; tanh bounds it
  to a turn command.
- `grow()` widens `fc1` at level-up: old columns copied exactly, new
  sensor columns start near zero — so the school never forgets.
- `act_batch()` steers all 32 mice in one matrix multiply. That's what
  makes 250 nightly episodes per generation run in under a minute.

## Follow one generation

Say the school is on Level 3 (`direction`) at generation 12:

```text
1. 50 brains: the top 20% cloned as-is, the rest bred by
   tournament + crossover + mutation.
2. Each brain drives all 32 mice through 5 fixed 1800-frame
   nights (same seeds every gen, so gains mean better genes).
3. Fitness = living mice x 2000 + mean frames lived (deaths dominate,
   time breaks ties).
4. Best score, death count, and population mean print to console;
   best-per-level and the full population snapshot save to disk.
5. One more mouse saved + 5 flat gens after 10 minimum? LEVEL UP: the next
   level must save one more mouse than this clearing average.
6. Otherwise: keep evolving. 15 slow gens trigger a shake-up
   (harder mutation + fresh immigrants, elites untouched).
   30 genuinely flat gens trigger autobar (the requirement drops to the
   whole mice this plateau already holds — time alone can never exit).
   to already-beaten fitness, announced loudly).
```

## Reading the console

```text
Level 1 (blind) | gen 0 | best 21.0/-- mice lost 11/32 (lost 14.4) (min 10 gens, stuck 0)
```

- `best 21.0/-- mice` — champion's mean survivors vs the mice required.
  L1 shows `--`: the baseline clears on plateau alone. Fitness still ranks
  brains behind the scenes (2000 per mouse + time), but only mice open
  doors — the same 21 survivors with better time can never advance a level.
- `lost 10/32` — what the best brain lost. **This is the number that matters:**
  it should fall level by level (~12 → ~10 → ~9 → … → 0).
- `mean 1431.3 (lost 13.2)` — whole-population average; rising means
  the school is converging, not just one lucky brain.
- `stuck 5` — gens since a real (+1) gain. The plateau clock: levels exit
  when this passes 5 with the gate earned, shakes fire at 15, autobar at 30.

Event lines: `*** LEVEL UP ***` (with `[wall]` = the next mice requirement),
`[shake]` (rescue diversity, nothing proven is risked), `[autobar]`
(requirement lowered to the whole mice the plateau already holds).

## Two ways to run

- `python main.py` — trains all levels headless (no window, console only)
  until the L6 finale gate (one more mouse past L5's clearing, max 32) clears.
  Leave it running; close anytime.
- `python main.py showcase` — the finished video: each level's all-time best
  replay with level-up cards in between. `SPACE` jumps ahead.
  `python main.py showcase 2 5` replays just those levels.
- Background: `MAX_GENS=200 nohup python main.py > train.log 2>&1 &`
- Resume after stopping (`Ctrl+C` is safe — every generation snapshots):
  `RESUME=1 python main.py` (bash) or
  `$env:RESUME = "1"; python main.py` (PowerShell).
- `[autobar]` — if a level sits 30+ genuinely flat gens, its requirement drops
  to the whole mice the plateau already averages (floor of the median,
  announced on console). A flat school certifies what it holds instead of
  grinding; time alone can never trigger this exit.

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

| Level | New sense | Wall |
|---|---|---|
| 1 · blind | nothing (just `bias`) | baseline, clears on plateau |
| 2 · proximity | how close the owl is (`dist`) | +1 mouse over L1's clearing |
| 3 · direction | where it's coming from (`dir x, dir y`) | +1 mouse over L2's clearing |
| 4 · intent | is it closing in (`closing`) | +1 mouse over L3's clearing |
| 5 · walls | where the edges are (`wall ↑↓→←`) | +1 mouse over L4's clearing |
| 6 · full sense | exact owl position (`aim x, aim y`) | +1 mouse over L5's clearing (finale) |

Only L1 has no requirement. Every later level must save one whole mouse more
than the previous clearing average (fractionals don't count), capped at 32.
The finale is simply the last such gate.

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

It unsticks itself: 15 slow generations triggers `[shake]`, 30 flat ones
triggers `[autobar]`, which lowers the mice requirement to what the plateau
already holds. Falling `lost-mean` in the console means it's learning. Interrupted? `RESUME=1 python main.py`
picks up where it died. Fresh physics change? Delete `checkpoints/` and
`logs/fitness.csv` and restart clean so old walls don't linger.
