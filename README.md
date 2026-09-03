# Brawl Stars Draft Agent

Recommends the next pick in a Brawl Stars ranked draft. A factorization machine
estimates win probability from team composition, map, and player skill; Monte
Carlo tree search plays the draft forward against a modelled opponent; and a
joint policy+value network trained by self-play replaces heuristic rollouts.

Data comes from a companion ETL pipeline —
[brawlstars-data-pipeline](https://github.com/elixf7/brawlstars-data-pipeline) —
which publishes ranked match telemetry as a versioned dataset.

> **Status.** The modelling code works and has produced trained models across
> three seasons. It is being turned into a reproducible pipeline: packaging,
> a training CLI, experiment tracking, and evaluation against baselines.

## What's here

```
src/bsdraft/
  data/       match loading, the empirical matchup database
  features/   sparse feature construction for the win-probability model
  fm/         the factorization machine: training, calibration, interpretation
  mcts/       draft state, tree search, rollouts, confidence, recommendation
  selfplay/   self-play generation, policy and joint policy+value networks
  pipeline/   stage orchestration
notebooks/    exploration and the current training entry points
figures/      calibration curves, brawler and context embeddings, map-skill affinity
tests/
```

## Setup

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Getting the data

Training reads from the published dataset rather than a local file, pinned to an
exact commit — a model trained against "the dataset" is not reproducible, because
the dataset grows every time the ETL pipeline runs.

```bash
uv run bsdraft-data seasons                    # what's available
uv run bsdraft-data resolve season53           # pin it to a commit
uv run bsdraft-data summary season53 --games   # load it as training does
```

```python
from bsdraft.data.prep import build_game_dataset
from bsdraft.data.sources import resolve_dataset

ref = resolve_dataset("season53")              # -> EliF77/...@4c45efee:season53
df, vocab = build_game_dataset(ref, elo_min=10, elo_max=23)
```

A local season database works the same way — pass a path instead of a ref, and
the same filters apply.

## Training

A run is described by a config file, not by remembered arguments:

```bash
uv run bsdraft-train show -c configs/default.toml   # what this run will do
uv run bsdraft-train fm   -c configs/default.toml   # train the evaluator
uv run bsdraft-train selfplay -c configs/default.toml
```

Each stage writes its artifacts and a manifest to `runs/<name>/`, recording the
seed, the git commit, the exact dataset revision, and the full resolved config —
so a result can be traced back to what produced it.

Runs are **bit-for-bit reproducible**: the same config and seed produce identical
weights. Seeding alone is not enough — reduction order stays free and drifts
weights by about 1e-8, close enough to look right and different enough that two
runs cannot be compared exactly — so strict deterministic algorithms are enabled
too.

## Comparing runs

Every run is recorded — its config, seed, git commit, dataset revision, metrics,
and the artifacts it produced with their content hashes.

```bash
uv run bsdraft-runs list                       # recent runs and headline metrics
uv run bsdraft-runs best --metric val_logloss  # the winner, and where its model is
uv run bsdraft-runs compare <run-a> <run-b>    # metrics side by side
```

`compare` also diffs the configs, so the reason two runs differ is on screen next
to the difference itself:

```
metric          181322-b0d7bf   181318-f1c952   181314-755698
val_auc                0.5032          0.5381          0.4821
val_logloss            0.6930          0.6929          0.6932

config differences
  fm.k                       64              32               8
```

```bash
uv run pytest
uv run ruff check src/ tests/
```

Training artifacts, feature matrices, and season databases are git-ignored:
they are reproducible from a config plus the published dataset.

## License

MIT — see [LICENSE](LICENSE). Not affiliated with or endorsed by Supercell; fan
content made under Supercell's
[Fan Content Policy](https://supercell.com/en/fan-content-policy/).
