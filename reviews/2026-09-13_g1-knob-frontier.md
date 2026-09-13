# G1 knob-frontier measurement (2026-09-13, after corpus saturation)

## Question
After the corpus-saturation pass (`2550aa2`, 138 techniques), does any
§8.2 selection-knob setting satisfy BOTH G1 halves — repeat-overlap < 0.30
AND ≥2 judge-plausible suggestions per problem on all five?

## Method
Monte Carlo over the warm full-field score cache (glm-5.3, 138 candidates,
all five canonical problems), 40 seeds per cell, exact `explore_methods`
code path. "Plausible" = a card the blind judge rated ≥3/5 for that problem
in ANY of rounds 1–6 (fixed historical ratings — no live-draw feedback).
Two sweeps: (1) `selection_pool_temp_multiplier` 1.5–6.0; (2)
`selection_min_eligible` 8–24.

## Results

Sweep 1 (min_eligible=24):

| temp | overlap mean/max | P(≥2 plausible)/problem | P(all 5) |
|---|---|---|---|
| 1.5 | 0.216 / 0.239 | [1.0, .47, .45, .12, .07] | 0.00 |
| 2.0 | 0.216 / 0.243 | [1.0, .42, .50, .15, .12] | 0.00 |
| 3.0 | 0.213 / 0.242 | [1.0, .42, .50, .15, .10] | 0.00 |
| 4.0 | 0.212 / 0.239 | [1.0, .40, .47, .10, .15] | 0.00 |
| 6.0 | 0.205 / 0.229 | [1.0, .40, .45, .10, .12] | 0.00 |

Sweep 2 (temp=3.0):

| min_eligible | overlap mean/max | P(≥2 plausible)/problem | P(all 5) |
|---|---|---|---|
| 8  | 0.563 / 0.715 (RED) | [1.0, 1.0, .33, .80, .00] | 0.00 |
| 12 | 0.388 / 0.457 (RED) | [1.0, .80, .33, .60, .00] | 0.00 |
| 16 | 0.298 / 0.346 | [1.0, .70, .28, .30, .15] | 0.01 |
| 24 | 0.213 / 0.242 | [1.0, .42, .50, .15, .10] | 0.00 |

## Reading

1. Temperature is not the lever: scores compress into ~[0.4, 0.9], so even
   an unflattened softmax differentiates the eligible set by barely e^0.3.
2. `min_eligible` IS the judge-half lever but moves overlap in lockstep:
   the frontier between the two halves runs from (overlap 0.21, P(all5)
   0.00) to (overlap 0.30, P(all5) 0.01). **No feasible point satisfies
   both.**
3. Tightening `min_eligible` can also *exclude* mid-ranked plausible cards
   (P5 collapses at 8–12: Kalman/CWT sit outside the top-8) — plausible-set
   membership and score rank are imperfectly correlated.
4. Empirical rounds agree: post-saturation round 6 drew [5,1,2,1,2] — two
   problems newly pass (P3, P5), total ≥3-rated picks rose 7 → 11, but P2
   and P4 landed at 1 in both independent samples, matching the simulated
   rates (~0.10–0.15 per sample).

## Conclusion

The G1 judge half, as specified (≥2 suggestions ≥3/5 on ALL five problems,
single blind draw per problem), is **not reachable by any in-scope §8.2
knob on this corpus with this judge's strictness** — a measured infeasibility,
not an absence of tuning effort. The delta review predicted this shape:
"judge variance … honestly cuts against ever passing a fixed-draw
criterion." Crossing it requires a protocol/plan decision (e.g. judging the
suggestion DISTRIBUTION or a consensus across draws — a plan §18.0 change,
Tier-3), or proceeding with G4 recorded red.
