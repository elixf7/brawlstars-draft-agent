# The model

What it predicts, how it's built, and what the numbers mean.

## The problem

Given six characters — three a side — plus the map, the mode, and roughly how
strong the lobby is, estimate the probability that team 1 wins.

That is all the information there is. The public API reports the finished draft,
not who picked when, and not what was banned. So the model sees *compositions*,
never *sequences*.

## What good looks like here

Matches are decided by execution as much as by composition. Even a perfect
read of the draft leaves most of the outcome undetermined, and the better
composition loses routinely. This is a domain where small, real edges are the
honest ceiling.

So the model is judged against predictors that know progressively more, all
scored on the same held-out games:

| Predictor | Knows | Log-loss | AUC | ECE |
| --- | --- | ---: | ---: | ---: |
| Constant | Nothing | 0.6931 | 0.500 | 0.001 |
| Character win rate | Which characters win | 0.6850 | 0.597 | 0.046 |
| Character × map | ...and where | 0.6794 | 0.624 | 0.057 |
| Head-to-head | Which beat which | 0.6834 | 0.612 | 0.056 |
| **This model** | All of it, jointly | **0.6598** | **0.644** | **0.005** |

Season 53: 1,051,296 games to train, 262,824 held out. The split is
**chronological** — the model is scored on matches played after everything it
learned from. A random split would leak the current meta backwards and flatter
the result.

Two readings matter. The model's lift over random (0.0333) is **more than double**
the best simple alternative's (0.0138). And its calibration error is an order of
magnitude smaller, which is not a footnote — see below.

## Why calibration, not just ranking

AUC asks whether the model *ranks* matchups correctly. Calibration asks whether
"65%" means 65%.

The recommender searches thousands of possible drafts and multiplies these
probabilities together as it goes. An evaluator that is confidently wrong
compounds its error at every ply; one that is honestly uncertain does not. The
counting baselines rank respectably (AUC 0.61–0.62) and are badly overconfident
(ECE 0.046–0.057) — they would be actively misleading inside a search.

## Architecture

A field-aware factorization machine, scored as a **difference between the two
teams**:

```
logit = own(team 1) - own(team 2)
      + counter(team 1 attacking team 2)
      - counter(team 2 attacking team 1)

where own(team) = strength + synergy + context
```

Each character gets several small vectors rather than one, because the roles are
different: playing *beside* a teammate, playing *against* an opponent, and
suiting a *map*. Counters use separate attack and defend vectors — an inner
product is symmetric, so a single vector per character would say the same thing
about "A beats B" as about "B beats A".

There is deliberately **no bias term**. A constant would survive the difference
and break the guarantee below. The slight team-1 win rate in the data is a
labelling artefact, not a property of either side.

### The guarantee

Because the score is a difference, `P(A beats B) + P(B beats A) = 1` holds
**exactly**, by construction — verified to 0 error in both the training and
inference paths.

This is not decoration. Tree search flips the evaluator's output to model the
opponent's perspective, and self-play labels the second team `1 − p`. Both
assume that identity.

### What the previous version did

The predecessor gave `t1_SHELLY` and `t2_SHELLY` **independent embeddings** and
learned symmetry approximately, through an augmentation that showed every game
twice with the teams swapped. Measured on real games, it never got there:

```
mean |p + p_flip − 1| = 0.047      max = 0.307
85% of games disagreed by more than 0.01
```

So the opponent model disagreed with the player model by up to 0.31, and every
self-play label inherited that error systematically. Making symmetry structural
fixed it, removed the need for augmentation (halving the rows, roughly 15×
faster to train), and improved every metric:

| | Log-loss | AUC | Brier | Symmetry error |
| --- | ---: | ---: | ---: | ---: |
| Original FM | 0.6631 | 0.6373 | 0.2354 | 0.047 mean / 0.307 max |
| Antisymmetric FFM | **0.6598** | **0.6441** | **0.2339** | **0** |

The original is kept behind `fm.model = "classic"` so earlier artifacts stay
reproducible.

## Where the signal is

Zeroing the interaction terms and keeping only per-character weights:

```
full model    log-loss 0.6608   AUC 0.641
linear only   log-loss 0.6896   AUC 0.581
```

Pairwise structure is worth **+0.06 AUC** — nearly the whole edge. Synergy and
counter effects are real, and a model without them is barely better than
counting.

## Choosing hyperparameters

The original FM was insensitive to them: widening `k` from 32 to 128 moved
log-loss by 0.0002, and training loss sat within 0.0003 of validation. It was
limited by its feature set, not its capacity, so tuning was pointless.

The field-aware model does respond, so the defaults were swept rather than
inherited:

| k | weight decay | lr | Log-loss | AUC |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 1e-5 | 3e-3 | 0.6603 | 0.6433 |
| 64 | 1e-4 | 3e-3 | 0.6601 | 0.6437 |
| **64** | **1e-5** | **1e-3** | **0.6598** | **0.6441** |
| 128 | 1e-5 | 3e-3 | 0.6615 | 0.6407 |

## Search and self-play

Tree search plays the remaining picks forward against a modelled opponent, so a
pick that looks strong in isolation but is easily answered gets discounted.
Self-play then distils that search into a network that proposes good picks
directly, which is what makes a recommendation fast enough to use mid-draft.

**This is distillation, not reinforcement learning.** The training signal is the
win-probability model's own opinion of the finished draft, not a match outcome.
No new information enters the loop, so it can learn to reach the same answers
more cheaply but cannot exceed what the evaluator already knows. Describing it
as RL would overstate it.

### What it actually buys, measured

Three iterations, 150 games each at 2,000 simulations per pick, 13.2 minutes:

| Iteration | Mean win probability vs. previous | Promoted? |
| ---: | ---: | --- |
| 0 | 0.5393 | yes |
| 1 | 0.4982 | no |
| 2 | 0.5008 | no |

The first policy beats searching without one. The next two are coin flips
against it. That is the shape the theory predicts: distillation captures the
search in one pass and then has nothing further to learn, because the evaluator
it is learning from has not changed.

So **one iteration is the useful one**, and the loop's value is speed — a
recommendation without a live tree search — rather than strength. Budgeting five
iterations spends four of them for nothing.

The promotion figure deserves a caveat that the original threshold hid. At 60
evaluation games, 0.5393 sits somewhere between 1.5 and 3 standard errors above
even depending on the spread of per-game probabilities. A bare "0.5393 > 0.525"
reads as decisive and is not, which is why `promotion_verdict` reports the
margin against its standard error rather than only the comparison.

There is a related trap, now closed. Promotion of a new policy used to be decided
by the *same* evaluator the policy was being optimised against — so a policy that
learned to draft compositions the evaluator overrates would be promoted for it,
with the number looking healthy the whole way. Two evaluators are now trained on
disjoint halves of the season: the first drives search, the second decides
promotion, and the verdict reports a margin against its standard error so a
narrow win can be told from noise.

## Limitations

- **No pick order, no bans.** Compositions only; the search must assume an order.
- **Not a uniform sample.** Games come from crawling outward from seed players
  within an elo band, so active and higher-rated players are over-represented.
- **One season per model.** Balance changes move what is strong, so seasons are
  not pooled.
- **Self-play is bounded by the evaluator**, as above.
