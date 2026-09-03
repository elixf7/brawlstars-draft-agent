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

| | |
| --- | --- |
| `src/` | Feature engineering, the FM evaluator, MCTS, self-play, and the joint network |
| `notebooks/` | Exploration and the current training entry points |
| `figures/` | Calibration curves, brawler and context embeddings, map–skill affinity |

## Setup

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Training artifacts, feature matrices, and season databases are git-ignored:
they are reproducible from a config plus the published dataset.

## License

MIT — see [LICENSE](LICENSE). Not affiliated with or endorsed by Supercell; fan
content made under Supercell's
[Fan Content Policy](https://supercell.com/en/fan-content-policy/).
