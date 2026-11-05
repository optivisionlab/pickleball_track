# Football (SoccerNet shots, 15 s clips): v3 / v4 / v5, goal / saved / off_target

Train = part3-6 (6 094 clips), valid = part1 (1 931), test = part2 (2 137); the shot is at 5 s. Inputs and hyper-parameters of each version were chosen on valid macro-F1 (run.sh lists them); test is only reported. Seeds 42, 43, 44, 45, 46: mean ± std over the seeds.

| Version | Valid macro-F1 | **Test macro-F1** | Test accuracy | F1 goal | F1 saved | F1 off_target | best epoch | trained params |
|---|---|---|---|---|---|---|---|---|
| **v3** | 0.8088 | **0.8073 ± 0.0022** | 0.7785 | 0.932 | 0.713 | 0.778 | 22 | 24.1M |
| **v4** | 0.8110 | **0.8068 ± 0.0068** | 0.7747 | 0.953 | 0.685 | 0.782 | 14 | 14.5M |
| **v5** | 0.7957 | **0.7864 ± 0.0053** | 0.7530 | 0.923 | 0.684 | 0.752 | 19 | 5.9M |
