### Cloth Move: cloth pixels, last frame

| test | n | real | LeWM probe, real frames | LeWM probe, 25-step | LeWM probe, rollout | DINO-WM Enc->Dec (#6) | DINO-WM 25-step (#6) | DINO-WM rollout (#6) | DINO-WM probe, real | DINO-WM probe, rollout |
|---|---|---|---|---|---|---|---|---|---|---|
| InD | 100 | 1,819 | 2,134 (117%) | 2,151 (118%) | 2,140 (118%) | 1,790 (98%) | 1,801 (99%) | 1,888 (104%) | 1,831 (101%) | 1,957 (108%) |
| OoD, 0.26x area | 50 | 404 | 552 (137%) | 849 (210%) | 744 (184%) | 388 (96%) | 840 (208%) | 1,129 (279%) | 422 (104%) | 1,240 (307%) |
| OoD, 2.5x area | 50 | 4,475 | 4,248 (95%) | 3,708 (83%) | 2,832 (63%) | 4,352 (97%) | 2,773 (62%) | 1,836 (41%) | 4,454 (100%) | 1,867 (42%) |

Probe on held-out real frames (all frames, all groups): R2 0.880, MAE 483 px.

| test | n | 1-step MSE | copy-last | ratio | rollout MSE | copy-last | ratio | rollout last step | copy-last |
|---|---|---|---|---|---|---|---|---|---|
| InD | 100 | 0.010 | 0.392 | 0.02 | 0.016 | 1.984 | 0.01 | 0.035 | 1.982 |
| OoD, 0.26x area | 50 | 0.021 | 0.227 | 0.09 | 0.227 | 1.220 | 0.19 | 0.183 | 1.511 |
| OoD, 2.5x area | 50 | 0.026 | 0.228 | 0.12 | 0.444 | 1.538 | 0.29 | 0.410 | 2.053 |

Embedding variance per dimension (real grid frames): 0.976. Weights: weights_epoch_100.pt.

### Push Sand: sand pixels, last frame

| test | n | real | LeWM probe, real frames | LeWM probe, 25-step | LeWM probe, rollout | DINO-WM Enc->Dec (#6) | DINO-WM 25-step (#6) | DINO-WM rollout (#6) |
|---|---|---|---|---|---|---|---|---|
| InD | 100 | 1,792 | 1,776 (99%) | 1,772 (99%) | 1,772 (99%) | 1,729 (96%) | 1,703 (95%) | 1,665 (93%) |
| OoD, 2x sand | 100 | 3,886 | 3,845 (99%) | 2,817 (72%) | 2,526 (65%) | 3,215 (83%) | 2,405 (62%) | 1,843 (47%) |

Probe on held-out real frames (all frames, all groups): R2 0.942, MAE 165 px.

| test | n | 1-step MSE | copy-last | ratio | rollout MSE | copy-last | ratio | rollout last step | copy-last |
|---|---|---|---|---|---|---|---|---|---|
| InD | 100 | 0.009 | 0.144 | 0.06 | 0.032 | 1.524 | 0.02 | 0.064 | 1.920 |
| OoD, 2x sand | 100 | 0.032 | 0.072 | 0.44 | 0.218 | 0.724 | 0.30 | 0.320 | 1.057 |

Embedding variance per dimension (real grid frames): 0.880. Weights: weights_epoch_115.pt.

### Stack Cube: stacked (yellow on green), share of failed episodes stacked in the last frame

| test | n | real | LeWM probe, real frames | LeWM probe, 25-step | LeWM probe, rollout | DINO-WM Enc->Dec (#6) | DINO-WM 25-step (#6) | DINO-WM rollout (#6) |
|---|---|---|---|---|---|---|---|---|
| InD, failed | 17 | 12% | 0% | 0% | 0% | 18% | 29% | 24% |
| OoD, failed | 71 | 8% | 0% | 0% | 0% | 6% | 14% | 23% |

Probe on held-out real frames (all frames, all groups): accuracy 0.843.

| test | n | 1-step MSE | copy-last | ratio | rollout MSE | copy-last | ratio | rollout last step | copy-last |
|---|---|---|---|---|---|---|---|---|---|
| InD | 67 | 0.004 | 0.004 | 1.19 | 0.041 | 0.020 | 2.04 | 0.090 | 0.041 |
| OoD | 271 | 0.003 | 0.003 | 1.22 | 0.040 | 0.023 | 1.72 | 0.072 | 0.031 |

Embedding variance per dimension (real grid frames): 1.008. Weights: weights_epoch_175.pt.

### Pour Water: water pixels in the box band, mean of frames 250-295

| test | n | real | LeWM probe, real frames | LeWM probe, 25-step | LeWM probe, rollout | DINO-WM Enc->Dec (#6) | DINO-WM 25-step (#6) | DINO-WM rollout (#6) |
|---|---|---|---|---|---|---|---|---|
| InD | 50 | 1,970 | 1,994 (101%) | 1,962 (100%) | 2,081 (106%) | 1,962 (100%) | 1,984 (101%) | 2,045 (104%) |
| OoD, less water | 26 | 1,384 | 1,386 (100%) | 1,421 (103%) | 1,922 (139%) | 1,370 (99%) | 1,593 (115%) | 2,049 (148%) |
| OoD, more water | 24 | 2,407 | 2,354 (98%) | 2,254 (94%) | 1,984 (82%) | 2,326 (97%) | 2,162 (90%) | 2,077 (86%) |

Probe on held-out real frames (all frames, all groups): R2 0.968, MAE 140 px.

| test | n | 1-step MSE | copy-last | ratio | rollout MSE | copy-last | ratio | rollout last step | copy-last |
|---|---|---|---|---|---|---|---|---|---|
| InD | 50 | 0.006 | 0.060 | 0.10 | 0.236 | 1.560 | 0.15 | 0.481 | 1.415 |
| OoD, less water | 26 | 0.007 | 0.053 | 0.13 | 0.351 | 1.481 | 0.24 | 0.738 | 1.315 |
| OoD, more water | 24 | 0.008 | 0.041 | 0.20 | 0.688 | 1.257 | 0.55 | 1.321 | 1.070 |

Embedding variance per dimension (real grid frames): 1.001. Weights: weights_epoch_33.pt.

