# Brawl Stars Draft Agent

**Which character should you pick next?**

Brawl Stars is a team game where two sides alternately pick three characters
each, before the match starts. Those six picks decide a lot: some characters
counter others, some work well together, and what's strong depends on the map.
Once the picking is over, you play the team you built.

This project learns which picks win, from 1.3 million real ranked games, and
recommends the next one.

[![CI](https://github.com/elixf7/brawlstars-draft-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/elixf7/brawlstars-draft-agent/actions/workflows/ci.yml)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Dataset](https://img.shields.io/badge/data-Hugging%20Face-yellow)](https://huggingface.co/datasets/EliF77/brawlstars-ranked)

---

## Why this is hard

You are predicting the winner of a match that hasn't been played, from nothing
but the six characters chosen and the map. No player skill history, no in-game
events — just the draft.

There is a ceiling on how well anyone can do that. Matches are decided by
execution as much as by composition, and the better team loses often. A model
that claimed 90% accuracy here would be broken, not brilliant. The honest
question is: **how much better than guessing can you get, and is it more than
you'd get from simply counting?**

## What it achieves

Every predictor below is scored on the same 262,824 games, held out **by time** —
so the model is judged on matches played after everything it learned from, the
way it would be used.

| Predictor | What it knows | Log-loss | AUC | Calibration error |
| --- | --- | ---: | ---: | ---: |
| Coin flip | Nothing | 0.6931 | 0.500 | 0.001 |
| Character win rates | Which characters win | 0.6850 | 0.597 | 0.046 |
| Character × map | ...and where they win | 0.6794 | 0.624 | 0.057 |
| Head-to-head rates | Which characters beat which | 0.6834 | 0.612 | 0.056 |
| **This model** | **All of it, jointly** | **0.6598** | **0.644** | **0.005** |

Lower log-loss is better; 0.6931 is what you score knowing nothing at all.

**The model captures more than twice as much signal as the best simple
alternative** — and that comparison is the result, not the raw number. Counting
character-and-map win rates already reaches 0.6794, so a model reporting "0.68"
would have been beaten by a lookup table.

**It also knows how confident to be.** The calibration column measures whether
"65% likely to win" actually happens 65% of the time. The counting methods rank
matchups reasonably but are badly overconfident. That distinction matters here
because the recommender searches thousands of possible drafts and multiplies
these probabilities together — an evaluator that is confidently wrong compounds
its error at every step, while one that is honestly uncertain does not.

## How it works

**1. Learn what wins.** A factorization machine over the six characters, the
map, the mode, and the skill level of the lobby. Rather than one weight per
character, it learns a small vector for each, so it can express that two
characters work well *together* or that one *counters* another — the pairwise
effects that make drafting interesting. Removing those interactions drops AUC
from 0.644 to 0.581, so that structure is where the value is.

The model scores a draft as the difference between the two teams, which
guarantees something you'd want to be true: its answer to "does A beat B" is
always exactly the inverse of "does B beat A". [The earlier version learned this
approximately and could disagree with itself by up to 0.31.](docs/MODEL.md)

**2. Search ahead.** A single pick isn't good or bad on its own — it depends on
what the opponent does next. Monte Carlo tree search plays the rest of the draft
forward against a modelled opponent, so a pick that looks strong but is easily
countered gets discounted.

**3. Learn to search less.** Self-play distills the search into a network that
proposes good picks directly, so the recommendation is fast enough to be useful
mid-draft.

## Where the data comes from

A companion project — **[brawlstars-data-pipeline](https://github.com/elixf7/brawlstars-data-pipeline)** —
crawls the Brawl Stars API on a schedule and publishes ranked matches as a
versioned dataset on [Hugging Face](https://huggingface.co/datasets/EliF77/brawlstars-ranked).

This repository reads from that dataset **pinned to an exact revision**. That
matters more than it sounds: the dataset grows every time the pipeline runs, so
"trained on the dataset" would not be reproducible. "Trained on commit `4c45efee`"
is.

```bash
uv run bsdraft-data seasons          # what's published
uv run bsdraft-data resolve season53 # pin it to a commit
```

## Running it

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run bsdraft-train fm -c configs/season53.toml   # ~20 seconds
uv run bsdraft-eval     -c configs/season53.toml   # the table above
```

A run is described by a config file, not by remembered arguments. Everything it
needs is in there — the data, the hyperparameters, the seed:

```toml
name = "season53"
seed = 20260903

[data]
season  = "season53"
elo_min = 10.0

[fm]
model = "ffm"
k     = 64
lr    = 1e-3
```

Runs are **bit-for-bit reproducible**: the same config and seed produce identical
weights. Seeding alone doesn't achieve that — floating-point reduction order
stays free and drifts weights by around 1e-8, close enough to look right and
different enough that two runs can't be compared — so strict deterministic
algorithms are enabled as well.

## Keeping track of experiments

Every run records what produced it: parameters, metrics, the git commit, the
dataset revision, and the artifacts it wrote with their content hashes.

```bash
uv run bsdraft-runs list                       # recent runs and their metrics
uv run bsdraft-runs best --metric val_logloss  # the winner, and where its model is
uv run bsdraft-runs compare <run-a> <run-b>
```

`compare` diffs the configurations as well, so *why* two runs differ appears next
to *how* they differ:

```
metric          181322-b0d7bf   181318-f1c952   181314-755698
val_auc                0.5032          0.5381          0.4821
val_logloss            0.6930          0.6929          0.6932

config differences
  fm.k                       64              32               8
```

## Layout

```
src/bsdraft/
  data/       loading matches, the head-to-head database
  features/   turning drafts into model inputs
  fm/         the win-probability model
  mcts/       draft state, tree search, recommendation
  selfplay/   self-play and the networks trained on it
  eval/       baselines, metrics, and the independent judge
configs/      run definitions
notebooks/    exploration
tests/        138 tests, no network required
```

## Honest limitations

- **Pick order is unrecoverable.** The API returns the six final characters with
  no record of who picked when, and no bans. The model sees compositions, not
  sequences; the search has to assume an order.
- **Self-play is distillation, not reinforcement learning.** Its training signal
  is the win-probability model's own opinion, not match outcomes, so it can
  learn to search more cheaply but cannot exceed what that model knows.
- **The sample is not uniform.** Matches are gathered by crawling outward from
  seed players within an elo band, so active and higher-rated players are
  over-represented.
- **One season at a time.** Balance changes shift what's strong, so models are
  trained per season rather than pooled.

## Documentation

| | |
| --- | --- |
| [`docs/MODEL.md`](docs/MODEL.md) | The model, why it's built this way, and what the numbers mean |
| [`docs/DATA.md`](docs/DATA.md) | Where the data comes from and what a row is |

## License

MIT — see [LICENSE](LICENSE). Not affiliated with or endorsed by Supercell; fan
content made under Supercell's
[Fan Content Policy](https://supercell.com/en/fan-content-policy/).
