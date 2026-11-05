# Football (SoccerNet shots, 15 s clips): v3 / v4 / v5, goal / saved (saved = saved + off_target)

Train = part3-6 (6 094 clips), valid = part1 (1 931), test = part2 (2 137); the shot is at 5 s. Inputs and hyper-parameters of each version were chosen on valid macro-F1 (run.sh lists them); test is only reported. Seeds 42, 43, 44, 45, 46: mean ± std over the seeds.

| Version | Valid macro-F1 | **Test macro-F1** | Test accuracy | F1 goal | F1 saved | best epoch | trained params |
|---|---|---|---|---|---|---|---|
| **v3** | 0.9755 | **0.9695 ± 0.0024** | 0.9842 | 0.948 | 0.991 | 16 | 24.1M |
| **v4** | 0.9748 | **0.9683 ± 0.0020** | 0.9833 | 0.946 | 0.990 | 28 | 14.5M |
| **v5** | 0.9646 | **0.9540 ± 0.0043** | 0.9760 | 0.922 | 0.986 | 25 | 9.5M |
