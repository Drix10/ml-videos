# Reference transcript (manual) — the video this project mirrors

Single narrator throughout. Timestamps are video time.

## Speaker: Narrator

**[00:00]**
Each of these fish has a neural network for a brain, but it's totally empty,
no senses, no training, so they just drift, and the predator picks them off.
Give every fish one input, how close the predator is, and they bolt when it
nears. But blind to direction, half still swim into its jaws. Add the
direction it's coming from, and they finally flee the right way, and we never
program this, the network evolves it on its own.

**[00:23]**
Feed it the predator's closing speed so it feels the lunge coming, cuts away
early and the shark overshoots and sails right past. Add wall sensors so they
stop cornering themselves and the survivors climb. Only the few still boxed
against an edge get caught. Full senses now and watch. Every single fish slips
the predator, smooth and untouchable, a brain that started with nothing and
taught itself to survive. Follow for more.

## Beat → our implementation

| Video beat | Our level | How it maps |
|---|---|---|
| Empty brain, drift, picked off | 1 · blind (`bias`) | No sensors; score = seconds lived |
| One input: closeness; bolt, but direction-blind | 2 · proximity (`dist`) | Flee magnitude only |
| Add direction; flee the right way; never programmed | 3 · direction (`dir x`, `dir y`) | GA discovers aimed fleeing |
| Closing speed; cut away; predator overshoots | 4 · intent (`closing`) | Owl turns wide (`OWL_TURN`), early cutaways win |
| Wall sensors; stop cornering | 5 · walls (4 sensors) | Margin bounces + corner awareness |
| Full senses; untouchable | 6 · full sense (`aim x`, `aim y`) | Ends on 1800: near-zero catches |

Our cast swap (owl/mice, gold-on-midnight) changes the skin, never the story:
fixed predator vs evolving prey, survival fitness, countdown counter.
