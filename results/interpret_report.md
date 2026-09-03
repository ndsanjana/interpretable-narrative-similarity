# Interpretability report

Benchmark: ATU-Union benchmark
Reference checkpoint: ckpt_seed111.pt
Triplets scored: 1346

## Channel decomposition (additive core)

Identity check: R^2 of the fused score on the two channel cosines = 1.000000000000 (max abs residual 3.141e-08). The additive core is an identity, not a fit.

| cell | n | share pct | fused accuracy |
| --- | --- | --- | --- |
| graph right text right | 953 | 70.80 | 100.00 |
| graph right text wrong | 96 | 7.13 | 94.79 |
| graph wrong text right | 233 | 17.31 | 28.33 |
| graph wrong text wrong | 64 | 4.75 | 0.00 |

Accuracy: graph-only 77.93, fused 82.47.
Mean graph contribution share of the fused margin: 0.7418.

## Channel angles

Decision space: the two per-triplet margin vectors sit at 48.83 degrees (centered 65.40, Pearson r 0.4162); the channels agree on the sign of the decision on 75.56 percent of triplets.

Representation space over 1381 shared tales, 16 components: mean principal angle 59.85 degrees, smallest 27.14, top canonical correlation 0.8899.

Type-disjoint linear probe (graph embedding -> text embedding): R^2 0.0967 on 255 held-out tales.

## Per-scorer breakdown

| scorer | overall |
| --- | --- |
| graph only | 77.93 |
| fusion additive core | 82.47 |
| fusion learned scalar | 83.06 |
| fusion gated | 82.17 |

## Gate analysis

Gate mean 0.5118 (sd 0.0064); the graph channel is favoured on 98.9 percent of scored pairs.


Departure from additivity: the gated score regresses on the two cosines with R^2 0.999731, so 0.03 percent of its variance is not expressible as fixed channel mixing.

## Causal edge knockout

Fused columns use alpha 0.4344 / beta 0.5656 on both the reference and every variant.

| variant | graph overall | delta | fused overall | delta |
| --- | --- | --- | --- | --- |
| inference knockout: causes | 78.23 | 0.30 | 82.76 | -0.30 |

