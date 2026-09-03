# Test fixtures

## `games_sample.parquet`

4,000 games sampled evenly across season 53, taken from the published dataset
(`EliF77/brawlstars-ranked`) through the same loading path training uses. Every
map and mode is represented, with the real brawler vocabulary, real elo and
`skill_ns` distributions, and real labels.

The modelling columns carry no player identifiers — the ETL export keeps
`star_player_tag` out of the training projection — so nothing needed
anonymising.

It exists because synthetic rows agree with whatever assumptions wrote them.
Feature construction is exactly where that matters: a symmetry augmentation that
swaps the wrong block, or an offset that overlaps two feature groups, produces a
matrix that is the right shape and the wrong content.

Rebuild it with:

```python
from bsdraft.data.prep import build_game_dataset
from bsdraft.data.sources import resolve_dataset

df, _ = build_game_dataset(resolve_dataset("season53"), elo_min=10, elo_max=23)
df.iloc[::len(df) // 4000].head(4000).reset_index(drop=True).to_parquet(
    "tests/fixtures/games_sample.parquet", compression="zstd", index=False)
```
