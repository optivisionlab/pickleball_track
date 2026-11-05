# NSVA basketball: v3 / v4 / v5 (final configuration)

5-fold cross-validation grouped by game (8 329 clips, 96 games), classes no_event / hit / miss. Seeds 1, 2, 42; per seed the score is the mean over the 5 test folds, the table shows mean ± std over the seeds. Val = inner validation games (used for early stopping).

| Version | Val macro-F1 | **Test macro-F1** | Test accuracy | F1 no_event | F1 hit | F1 miss | seed 1 | seed 2 | seed 42 |
|---|---|---|---|---|---|---|---|---|---|
| **v3** | 0.8471 | **0.8319 ± 0.0035** | 0.8325 | 0.7916 | 0.8876 | 0.8165 | 0.8279 | 0.8341 | 0.8337 |
| **v4** | 0.8453 | **0.8282 ± 0.0009** | 0.8292 | 0.7879 | 0.8814 | 0.8154 | 0.8293 | 0.8275 | 0.8280 |
| **v5** | 0.8069 | **0.7959 ± 0.0007** | 0.7962 | 0.7686 | 0.8349 | 0.7841 | 0.7966 | 0.7955 | 0.7954 |

Configuration (train.py defaults): VideoMAE self-supervised, mean + 2x2 grid tokens, + second-half view (--tail), BEATs base, fixed text prompt (v4 / v5), fusion 384, weight decay 0.05, token-drop 0.1, audio-dropout 0.2; 50 epochs, patience 10, batch 256, lr 3e-4, label smoothing 0.1.
