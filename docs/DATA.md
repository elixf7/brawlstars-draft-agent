# The data

## Where it comes from

A companion project,
[brawlstars-data-pipeline](https://github.com/elixf7/brawlstars-data-pipeline),
crawls the public Brawl Stars API on a schedule and publishes ranked matches as
a versioned dataset on
[Hugging Face](https://huggingface.co/datasets/EliF77/brawlstars-ranked).

There is no bulk endpoint — the API only returns a given player's recent matches
— so that pipeline assembles the dataset by walking the player graph outward
from seed players, deduplicating as it goes.

## Pinning

Training reads the dataset **at an exact commit**:

```python
from bsdraft.data.sources import resolve_dataset
ref = resolve_dataset("season53")     # EliF77/brawlstars-ranked@4c45efee:season53
```

This matters because the dataset grows every time the pipeline runs. A model
"trained on the dataset" is not reproducible; a model trained on a named commit
is. Resolving without specifying a revision still records whichever commit was
current, so a run that wasn't pinned in advance can still be reproduced after
the fact.

A local season database can be used instead — pass a path rather than a ref, and
the same filters apply either way.

## A row

The published dataset stores one row per **set**: up to three games on one map,
first team to two wins. Training expands those into individual games, because
that is the unit being predicted.

| Column | Meaning |
| --- | --- |
| `battle_time` | When the set ended, UTC |
| `mode`, `map` | Fixed for the whole set |
| `t1_b{0,1,2}_name` | Team 1's three characters |
| `t2_b{0,1,2}_name` | Team 2's three characters |
| `avg_elo` | Mean rating across the six players |
| `skill_ns` | `avg_elo` normalised against the rating distribution *local in time* |
| `team1_wins` | The label, after expansion to games |

### Why `skill_ns` and not `avg_elo`

Ranked ratings reset at the start of each season and re-spread over the
following weeks. A rating of 16 in week one describes a very different match
from a 16 in week three. Training on the raw number would mean learning a moving
target.

`skill_ns` converts each match's average rating into its percentile *within a
three-day window*, then maps that onto a symmetric scale. It is comparable
across the whole season, which the raw rating is not. The ETL pipeline computes
it and flags rows whose time window had too few samples to trust; training drops
those.

## Filters applied

- `skill_ns_ok = 1` — the skill score came from a well-populated time window
- `avg_elo` within the configured band (10–23 by default)
- Incomplete teams dropped; draws excluded when expanding sets into games

## Scale

Season 53, as of writing: **571,123 sets → 1,314,120 games**, across 6 modes,
26 maps, and 106 characters.

The character vocabulary grows — it was 95 two seasons earlier. New characters
ship regularly, so the model and every baseline fall back gracefully on ones
they have not seen rather than failing.
