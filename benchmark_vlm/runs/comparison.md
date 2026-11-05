# NSVA made / miss / no-event: custom models vs zero-shot VLMs

_Generated 2026-10-21 by `make_report.py`. 8329 test clips, 96 games, 5 folds `StratifiedGroupKFold` by game (seed 42); every model is scored on the same test clips of each fold._

- **Custom v3/v4/v5**: trained per fold; the numbers are the test results of the best epoch (early-stopped on the fold's val split), from `weights_basketball/<v>/fold_k/result.json`.
- **VLM**: zero-shot, no training; the frame config was picked once on `val_fold1` (16f/448, 32f/448, 16f/640), then run on all test clips.
- **Params**: counted at inference from the real weights. Custom = pretrained encoders (VideoMAE-base 88.6M, BEATs 90.3M for v3/v4, CLIP text 63.2M for v4/v5) + the trained head.
- **Mean ± std**: over the 5 folds (population std, as in the training logs).
- **Δ vs vX**: mean macro-F1 of the row minus that of custom vX. `*` = significant (McNemar exact test on the per-clip predictions, p < 0.05).

## Summary

| # | Model | Type | Params (trained) | Macro-F1 mean ± std | Acc mean ± std | Δ vs v3 | Δ vs v4 | Δ vs v5 | F1 made | F1 miss | F1 no-event | Clips/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **custom v4** | custom (trained) | 265M (23M) | 0.8158 ± 0.0083 | 0.8169 ± 0.0088 | +0.001 | – | +0.069* | 0.868 | 0.806 | 0.774 | – |
| 2 | custom v3 | custom (trained) | 197M (18M) | 0.8150 ± 0.0093 | 0.8155 ± 0.0083 | – | -0.001 | +0.068* | 0.862 | 0.801 | 0.783 | – |
| 3 | custom v5 | custom (trained) | 169M (17M) | 0.7469 ± 0.0102 | 0.7466 ± 0.0103 | -0.068* | -0.069* | – | 0.766 | 0.737 | 0.737 | – |
| 4 | qwen3.5-9b | VLM zero-shot | 9.41B (0) | 0.6212 ± 0.0108 | 0.6257 ± 0.0096 | -0.194* | -0.195* | -0.126* | 0.778 | 0.574 | 0.512 | 0.86 |
| 5 | qwen3-vl-8b | VLM zero-shot | 8.77B (0) | 0.4424 ± 0.0079 | 0.5446 ± 0.0100 | -0.373* | -0.373* | -0.304* | 0.682 | 0.616 | 0.030 | 1.06 |
| 6 | internvl3.5-8b | VLM zero-shot | 8.53B (0) | 0.4286 ± 0.0106 | 0.5335 ± 0.0123 | -0.386* | -0.387* | -0.318* | 0.638 | 0.618 | 0.030 | 1.01 |
| 7 | qwen3-vl-4b | VLM zero-shot | 4.44B (0) | 0.4190 ± 0.0084 | 0.5223 ± 0.0090 | -0.396* | -0.397* | -0.328* | 0.621 | 0.607 | 0.030 | 1.77 |
| 8 | internvl3.5-4b | VLM zero-shot | 4.73B (0) | 0.3654 ± 0.0108 | 0.4796 ± 0.0123 | -0.450* | -0.450* | -0.381* | 0.489 | 0.590 | 0.018 | 1.35 |
| 9 | minicpm-v-4 | VLM zero-shot | 4.06B (0) | 0.3531 ± 0.0156 | 0.4583 ± 0.0155 | -0.462* | -0.463* | -0.394* | 0.420 | 0.579 | 0.060 | 1.28 |
| 10 | qwen3.5-4b | VLM zero-shot | 4.54B (0) | 0.2375 ± 0.0140 | 0.3612 ± 0.0123 | -0.578* | -0.578* | -0.509* | 0.508 | 0.001 | 0.205 | 1.40 |
| 11 | gemma4-e4b | VLM zero-shot | 8.00B (0) | 0.1839 ± 0.0079 | 0.3101 ± 0.0094 | -0.631* | -0.632* | -0.563* | 0.088 | 0.002 | 0.462 | 2.55 |

F1 per class is pooled over the 5 test folds. Clips/s: VLM inference on one A30 (video decoding included); custom models run on cached features, so no comparable number.

## Per fold

Macro-F1 / accuracy on each test fold (custom: best epoch of that fold).

| Model | Config | Fold 1 F1 / Acc | Fold 2 F1 / Acc | Fold 3 F1 / Acc | Fold 4 F1 / Acc | Fold 5 F1 / Acc | Mean F1 | Std F1 | Mean Acc | Std Acc |
|---|---|---|---|---|---|---|---|---|---|---|
| custom v4 | best epoch / fold | 0.8250 / 0.8275 | 0.8205 / 0.8203 | 0.8005 / 0.8008 | 0.8153 / 0.8177 | 0.8178 / 0.8181 | 0.8158 | 0.0083 | 0.8169 | 0.0088 |
| custom v3 | best epoch / fold | 0.8236 / 0.8234 | 0.8288 / 0.8273 | 0.8088 / 0.8104 | 0.8080 / 0.8103 | 0.8058 / 0.8059 | 0.8150 | 0.0093 | 0.8155 | 0.0083 |
| custom v5 | best epoch / fold | 0.7421 / 0.7419 | 0.7561 / 0.7560 | 0.7297 / 0.7291 | 0.7491 / 0.7489 | 0.7573 / 0.7570 | 0.7469 | 0.0102 | 0.7466 | 0.0103 |
| qwen3.5-9b | 32f_448 | 0.6256 / 0.6320 | 0.6335 / 0.6352 | 0.6123 / 0.6184 | 0.6296 / 0.6323 | 0.6049 / 0.6105 | 0.6212 | 0.0108 | 0.6257 | 0.0096 |
| qwen3-vl-8b | 32f_448 | 0.4406 / 0.5446 | 0.4572 / 0.5633 | 0.4335 / 0.5365 | 0.4387 / 0.5359 | 0.4421 / 0.5427 | 0.4424 | 0.0079 | 0.5446 | 0.0100 |
| internvl3.5-8b | 16f_448 | 0.4218 / 0.5304 | 0.4458 / 0.5533 | 0.4357 / 0.5395 | 0.4171 / 0.5169 | 0.4227 / 0.5275 | 0.4286 | 0.0106 | 0.5335 | 0.0123 |
| qwen3-vl-4b | 16f_640 | 0.4138 / 0.5192 | 0.4330 / 0.5386 | 0.4185 / 0.5221 | 0.4219 / 0.5206 | 0.4080 / 0.5110 | 0.4190 | 0.0084 | 0.5223 | 0.0090 |
| internvl3.5-4b | 16f_448 | 0.3718 / 0.4849 | 0.3831 / 0.4997 | 0.3624 / 0.4791 | 0.3554 / 0.4653 | 0.3546 / 0.4689 | 0.3654 | 0.0108 | 0.4796 | 0.0123 |
| minicpm-v-4 | 16f_448 | 0.3421 / 0.4483 | 0.3735 / 0.4826 | 0.3521 / 0.4528 | 0.3310 / 0.4389 | 0.3671 / 0.4689 | 0.3531 | 0.0156 | 0.4583 | 0.0155 |
| qwen3.5-4b | 16f_640 | 0.2595 / 0.3786 | 0.2312 / 0.3477 | 0.2451 / 0.3678 | 0.2335 / 0.3653 | 0.2180 / 0.3468 | 0.2375 | 0.0140 | 0.3612 | 0.0123 |
| gemma4-e4b | 16f_448 | 0.1724 / 0.2983 | 0.1863 / 0.3123 | 0.1781 / 0.3002 | 0.1949 / 0.3217 | 0.1877 / 0.3181 | 0.1839 | 0.0079 | 0.3101 | 0.0094 |

## Notes

- v3 and v4 are tied (difference not significant); every VLM is significantly below all three custom models.
- Most VLMs almost never answer `no-event` (F1 no-event ≈ 0.03), which caps their macro-F1 near 0.44.
- VLM parameter counts exclude Qwen3.5's MTP weights (not loaded at inference); Gemma 4 E4B includes its audio tower and per-layer embeddings (all loaded).
- `val_fold1` (used only to pick the VLM frame config) overlaps the test clips of folds 2–5; fold 1 is fully clean for the VLMs.
- Not included: minicpm-v-4.5, ref_sequential_16f_448 (stopped / not a model).
