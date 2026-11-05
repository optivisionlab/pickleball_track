# UCF-101: v3 / v4 / v5

Top-1 / macro-F1 (%) averaged over the 3 official splits (± std over the splits); train / test only on UCF-101, Kinetics is only the pre-training data of the backbone.

| Model | Top-1 | Top-5 | Macro-F1 | Mean class acc. |
|---|---|---|---|---|
| **v3** (frozen VideoMAE-K400) | **98.1 ± 0.7** | 99.9 | 98.1 | 98.2 |
| **v4** (frozen VideoMAE-K400) | **97.8 ± 0.6** | 99.9 | 97.8 | 97.9 |
| **v5** (frozen VideoMAE-K400) | **97.9 ± 0.5** | 99.8 | 97.9 | 98.0 |
| **v3** (VideoMAE-K400 fine-tuned end-to-end, 3/3 splits) | **98.1 ± 0.6** | 99.8 | 98.1 | 98.2 |

Frozen: VideoMAE-K400 (mean + 2x2 grid, 6 views: random view in training, mean of the 6 at test), CLIP ViT-B/16 image embedding, BEATs fine-tuned on AudioSet, class-description text for v4 / v5; 3 training groups held out to pick the epoch, then refit on the full training list.
Fine-tuned: 16 frames (TSN sampling), 25 epochs, layer-wise lr decay 0.75, cached BEATs audio, test = 2 temporal offsets x 3 crops.
