# Pickleball hit / miss / no-event: custom models vs zero-shot VLMs

_Generated 2026-10-22 by `make_report_pickleball.py`. 4685 test clips (1859 hit, 967 miss, 1859 no-event), 115 videos, 17 players, 5 folds `StratifiedGroupKFold` by player (seed 42); every model is scored on the same test clips of each fold._

- **Custom v3/v4/v5**: trained per fold (best epoch, early-stopped on the fold's val split); the test predictions of `weights_pickleball/<v>` are mapped onto each ~1 s clip (`make_splits_pickleball.py`) and scored here like the VLMs.
- **VLM**: zero-shot, no training; the frame config was picked once on `val_fold1` (16f/448, 32f/448, 16f/640), then run on all test clips.
- **Params**: counted at inference from the real weights. Custom = pretrained encoders (VideoMAE-base Kinetics 88.6M, BEATs 90.3M for v3/v4, CLIP text 63.4M for v4/v5) + the trained head.
- **Mean ± std**: over the 5 folds (population std, as in the training logs).
- **Δ vs vX**: mean macro-F1 of the row minus that of custom vX. `*` = significant (McNemar exact test on the per-clip predictions, p < 0.05).
- **made = hit**: the column names follow the NSVA report; for pickleball `made` is a hit.

## Summary

| # | Model | Type | Params (trained) | Macro-F1 mean ± std | Acc mean ± std | Δ vs v3 | Δ vs v4 | Δ vs v5 | F1 made | F1 miss | F1 no-event | Clips/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **custom v4** | custom (trained) | 246M (4M) | 0.9394 ± 0.0445 | 0.9428 ± 0.0377 | +0.011* | – | +0.084* | 0.960 | 0.936 | 0.932 | – |
| 2 | custom v3 | custom (trained) | 182M (3M) | 0.9284 ± 0.0564 | 0.9347 ± 0.0496 | – | -0.011* | +0.073* | 0.952 | 0.908 | 0.935 | – |
| 3 | custom v5 | custom (trained) | 155M (3M) | 0.8555 ± 0.1107 | 0.8617 ± 0.1076 | -0.073* | -0.084* | – | 0.888 | 0.835 | 0.861 | – |
| 4 | internvl3.5-8b | VLM zero-shot | 8.53B (0) | 0.7740 ± 0.1066 | 0.8047 ± 0.0933 | -0.154* | -0.165* | -0.082* | 0.791 | 0.666 | 0.888 | 0.48 |
| 5 | qwen3.5-9b | VLM zero-shot | 9.41B (0) | 0.7314 ± 0.0788 | 0.7429 ± 0.0695 | -0.197* | -0.208* | -0.124* | 0.669 | 0.740 | 0.800 | 0.88 |
| 6 | qwen3.5-4b | VLM zero-shot | 4.54B (0) | 0.6610 ± 0.0829 | 0.6919 ± 0.0700 | -0.267* | -0.278* | -0.194* | 0.619 | 0.580 | 0.795 | 1.46 |
| 7 | qwen3-vl-8b | VLM zero-shot | 8.77B (0) | 0.6035 ± 0.0430 | 0.6448 ± 0.0408 | -0.325* | -0.336* | -0.252* | 0.424 | 0.577 | 0.814 | 2.31 |
| 8 | internvl3.5-4b | VLM zero-shot | 4.73B (0) | 0.5933 ± 0.0502 | 0.6376 ± 0.0398 | -0.335* | -0.346* | -0.262* | 0.538 | 0.346 | 0.901 | 0.63 |
| 9 | qwen3-vl-4b | VLM zero-shot | 4.44B (0) | 0.5785 ± 0.0549 | 0.6264 ± 0.0542 | -0.350* | -0.361* | -0.277* | 0.351 | 0.567 | 0.822 | 1.60 |
| 10 | minicpm-v-4 | VLM zero-shot | 4.06B (0) | 0.4802 ± 0.0437 | 0.5553 ± 0.0542 | -0.448* | -0.459* | -0.375* | 0.090 | 0.483 | 0.870 | 1.28 |
| 11 | gemma4-e4b | VLM zero-shot | 8.00B (0) | 0.4181 ± 0.0288 | 0.5670 ± 0.0349 | -0.510* | -0.521* | -0.437* | 0.500 | 0.034 | 0.730 | 2.58 |

F1 per class is pooled over the 5 test folds. Clips/s: VLM inference on one A30 (video decoding included); custom models run on cached features, so no comparable number.

## Per fold

Macro-F1 / accuracy on each test fold (custom: best epoch of that fold).

| Model | Config | Fold 1 F1 / Acc | Fold 2 F1 / Acc | Fold 3 F1 / Acc | Fold 4 F1 / Acc | Fold 5 F1 / Acc | Mean F1 | Std F1 | Mean Acc | Std Acc |
|---|---|---|---|---|---|---|---|---|---|---|
| custom v4 | best epoch / fold | 0.9803 / 0.9812 | 0.9358 / 0.9415 | 0.8555 / 0.8721 | 0.9552 / 0.9540 | 0.9702 / 0.9653 | 0.9394 | 0.0445 | 0.9428 | 0.0377 |
| custom v3 | best epoch / fold | 0.9720 / 0.9732 | 0.8689 / 0.8927 | 0.8528 / 0.8605 | 0.9573 / 0.9570 | 0.9911 / 0.9899 | 0.9284 | 0.0564 | 0.9347 | 0.0496 |
| custom v5 | best epoch / fold | 0.9413 / 0.9465 | 0.8359 / 0.8548 | 0.6472 / 0.6558 | 0.9218 / 0.9219 | 0.9311 / 0.9295 | 0.8555 | 0.1107 | 0.8617 | 0.1076 |
| internvl3.5-8b | 32f_448 | 0.8547 / 0.8771 | 0.7220 / 0.7822 | 0.5888 / 0.6326 | 0.8294 / 0.8478 | 0.8749 / 0.8837 | 0.7740 | 0.1066 | 0.8047 | 0.0933 |
| qwen3.5-9b | 32f_448 | 0.7735 / 0.7820 | 0.7125 / 0.7270 | 0.5886 / 0.6163 | 0.7647 / 0.7728 | 0.8175 / 0.8166 | 0.7314 | 0.0788 | 0.7429 | 0.0695 |
| qwen3.5-4b | 16f_640 | 0.6808 / 0.7066 | 0.6609 / 0.6891 | 0.5118 / 0.5698 | 0.6850 / 0.7067 | 0.7666 / 0.7875 | 0.6610 | 0.0829 | 0.6919 | 0.0700 |
| qwen3-vl-8b | 16f_448 | 0.6113 / 0.6571 | 0.6112 / 0.6403 | 0.5215 / 0.5686 | 0.6261 / 0.6777 | 0.6471 / 0.6801 | 0.6035 | 0.0430 | 0.6448 | 0.0408 |
| internvl3.5-4b | 32f_448 | 0.6336 / 0.6700 | 0.6139 / 0.6381 | 0.4956 / 0.5628 | 0.6248 / 0.6727 | 0.5986 / 0.6443 | 0.5933 | 0.0502 | 0.6376 | 0.0398 |
| qwen3-vl-4b | 32f_448 | 0.5893 / 0.6422 | 0.5085 / 0.5645 | 0.5218 / 0.5605 | 0.6294 / 0.6857 | 0.6436 / 0.6790 | 0.5785 | 0.0549 | 0.6264 | 0.0542 |
| minicpm-v-4 | 16f_448 | 0.5061 / 0.5946 | 0.4828 / 0.5341 | 0.3996 / 0.4674 | 0.5285 / 0.6256 | 0.4840 / 0.5548 | 0.4802 | 0.0437 | 0.5553 | 0.0542 |
| gemma4-e4b | 16f_448 | 0.4505 / 0.6056 | 0.4290 / 0.5796 | 0.4047 / 0.5547 | 0.3688 / 0.5055 | 0.4375 / 0.5895 | 0.4181 | 0.0288 | 0.5670 | 0.0349 |

## Notes

- v4 is slightly but significantly above v3 (+0.011, McNemar p = 0.016), unlike NSVA where they tie; every VLM is significantly below all three custom models.
- Unlike NSVA, the VLMs do answer `no-event` (F1 0.73–0.90); their weak class is hit or miss (minicpm-v-4 hit F1 0.09, gemma4-e4b miss F1 0.03).
- Fold 3 (players Cuong, Son, Tu) is the hardest fold for the custom models and most VLMs, hence the large fold std.
- The custom numbers here are per clip and differ from `weights_pickleball/<v>/fold_k/result.json`, which scores 1 s training windows.
- VLM parameter counts exclude Qwen3.5's MTP weights (not loaded at inference); Gemma 4 E4B includes its audio tower and per-layer embeddings (all loaded).
- `val_fold1` (players Cuong, VanAnh; used only to pick the VLM frame config) overlaps the test clips of folds 3 and 4; folds 1, 2 and 5 are fully clean for the VLMs.
- Not included: none (all 8 VLMs finished).
