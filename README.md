<h1 align="center">ml-videos</h1>

<p align="center">
  <strong>A lab for machine-learning videos.</strong><br />
  Small brains, visible struggle, big learning curves — real training runs rendered as watchable videos.
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-Tiny%20Brains-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img alt="Portrait" src="https://img.shields.io/badge/Format-9%3A16%20Portrait-6C5CE7?style=for-the-badge" />
  <img alt="GitHub" src="https://img.shields.io/badge/GitHub-Drix10%2Fml--videos-181717?style=for-the-badge&logo=github&logoColor=white" />
</p>

<p align="center">
  <a href="https://github.com/Drix10/ml-videos">github.com/Drix10/ml-videos</a> · <a href="https://drix10.com">drix10.com</a>
</p>

## The Idea

The best ML videos share one shape: **you watch something dumb become something smart.** A school of fish that can't dodge, then can't be touched. A creature that flails, then walks. The learning curve *is* the content.

This repo exists to manufacture that shape for real — not animations pretending to learn, but actual training runs with real brains, where the video is the honest output of the science:

```text
pick a struggle -> build the smallest brain that can learn it
  -> train it on camera (logs + checkpoints) -> replay the learning as video
```

Every project here follows that pipeline. Same discipline, different worlds.

## The Formula

Each project is a self-contained folder that obeys five rules:

1. **One visible struggle.** A predator and prey, a maze, a race — something a viewer grasps in three seconds.
2. **The tiniest brain that can learn it.** Single-digit hidden units, shared across agents. If the brain is big, the video is boring; the struggle must live in the learning, not the parameters.
3. **Stages the eye can see.** Senses, obstacles, or opponents added level by level, so episode N of the video looks dumber than episode N+1. Checkpoints saved per stage.
4. **Training and rendering stay separate.** Headless trainer writes logs + checkpoints; a replay mode renders saved brains cinematically. One run produces both the science and the story.
5. **Nothing is faked.** Fixed evaluation seeds, honest death counts, no scripted behavior. If the brain didn't learn it, the video doesn't show it.

## Projects

| Project | Struggle | Brain | Status |
|---|---|---|---|
| [`night_hunt`](night_hunt/) | An owl hunts 32 mice sharing one evolving brain; six senses from blind to full | 9-unit MLP, genetic algorithm | Training → video next |

Future projects take the same shape in new worlds — anything in this spirit: foragers learning a map, racers learning a track, a swarm learning formation. If it can be learned by a tiny brain and *seen* being learned, it belongs here.

## How a New Project Gets Made

```text
1. Name the struggle in one sentence a viewer understands.
2. Fix the physics first, then never touch it again mid-project.
3. Design the curriculum: what does the final brain know that the first one doesn't?
4. Build the smallest brain + the fastest headless trainer that can teach it.
5. Add failsafes so training ends on its own: plateau gates, rescue shakes, honest bars.
6. Train once, fully. Watch the numbers, not the screen.
7. Replay the checkpoints, record the window, cut the video.
```

Steps 1–3 decide whether the video works. Steps 4–5 decide whether training finishes. Steps 6–7 are patience plus OBS.

## Try the Current One

```powershell
cd night_hunt
python -m pip install -r requirements.txt
python main.py             # train (headless)
python main.py showcase    # replay (record this)
```

Full console guide, failsafes, and tuning live in [`night_hunt/README.md`](night_hunt/README.md).

## Repository Map

| Path | Role |
|---|---|
| `night_hunt/` | Project 01: owl vs 32 mice, shared evolving brain, six senses |
| `night_hunt/README.md` | That project's deep dive |
| *future folders* | One folder per video, each obeying the formula above |

## License

All original code in this repository is MIT Licensed. Built for transparent experimentation and learning.
