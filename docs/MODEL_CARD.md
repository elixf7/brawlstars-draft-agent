# Model card — Brawl Stars draft evaluator

## What it does

Estimates the probability that team 1 wins a Brawl Stars ranked game, given the
six characters drafted, the map, the mode, and the skill level of the lobby. It
is the evaluator inside a draft recommender: tree search calls it thousands of
times to decide which character to pick next.

**Version:** `ffm` (antisymmetric field-aware factorization machine)
**Trained on:** season 53, `EliF77/brawlstars-ranked@4c45efee`
**Size:** 29,354 parameters

## Intended use

Suggesting picks during a ranked draft, and analysing which characters work
together or counter each other. It is a hobby project about a video game.

**Not intended for** anything outside this game, for judging individual players,
or as a basis for wagering. It predicts a noisy outcome from partial information
and is wrong often — see below.

## How it was trained

1,051,296 games, held out **chronologically**: trained on the earlier 80% and
scored on the later 20%, so it is judged on matches played after everything it
learned from. A random split would leak the current meta backwards and make the
result look better than it is.

Rows are filtered to matches whose skill score came from a well-populated time
window, and to an elo band of 10–23. Draws are excluded. Every run is seeded and
bit-for-bit reproducible; the exact configuration lives in
`configs/season53.toml`.

## How it performs

![What each approach is worth](../figures/baseline_comparison.png)

| Predictor | Log-loss | AUC | Calibration error |
| --- | ---: | ---: | ---: |
| Coin flip | 0.6931 | 0.500 | 0.001 |
| Character win rates | 0.6850 | 0.597 | 0.046 |
| Character × map | 0.6794 | 0.624 | 0.057 |
| Head-to-head rates | 0.6834 | 0.612 | 0.056 |
| **This model** | **0.6598** | **0.644** | **0.005** |

Read as a comparison, not a number. Counting character-and-map win rates already
reaches 0.6794, so a model reporting "0.68" would be beaten by a lookup table.
This one captures **more than twice** the signal of the best simple alternative.

Calibration matters as much as ranking here. The recommender multiplies these
probabilities across a search, so an evaluator that is confidently wrong
compounds its error at every step. The counting baselines rank respectably and
are badly overconfident; this model is an order of magnitude better calibrated.

## How much data it needs

![More data still helps](../figures/learning_curve.png)

Performance is still improving at a million games — there is no plateau. Two
practical consequences: retraining as data accumulates is worth doing, and a
model trained on less than about 100,000 games is **worse than counting** and
should not be shipped.

## Known limitations

- **It only sees the finished draft.** The API reports which six characters were
  picked, not in what order, and not what was banned. Sequence-dependent
  reasoning has to be inferred.
- **The outcome is mostly not about the draft.** Matches are decided by execution
  too. An AUC of 0.64 is a real edge on a genuinely noisy target, not a strong
  classifier — the better composition loses routinely.
- **The sample is not uniform.** Games are gathered by crawling outward from seed
  players inside an elo band, so active and higher-rated players are
  over-represented. It describes that population, not everyone.
- **One season only.** Balance changes move what is strong, so a model trained on
  one season should not be applied to another.
- **New characters degrade it.** The roster grew from 95 to 106 across the
  seasons observed. Unseen characters fall back to a default rather than
  failing, but predictions involving them are weaker.

## Guarantees it does hold

`P(A beats B) + P(B beats A) = 1`, exactly, by construction. The score is a
difference between the two teams, so no bias or asymmetry can survive. The
previous version learned this approximately and disagreed with itself by up to
0.307, which mattered because search flips the evaluator to model the opponent.

## Ethical considerations

The training data contains no player identifiers — the pipeline's modelling
projection excludes them. Nothing here profiles individuals. The dataset is
public game telemetry gathered through Supercell's official API under their fan
content policy.
