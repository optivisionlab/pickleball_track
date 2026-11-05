"""UCF-101 training parameters, chosen from analysis/ucf101_analysis.json (analyze_ucf101.py).

Dataset facts that drive the choices
    13,320 clips, 101 classes, 25 groups; official splits: ~9.5k train / ~3.7k test, test groups
    g01-07 / g08-14 / g15-21, no group overlap.
    72-124 train clips per class (balanced) -> plain cross-entropy, no class weights; top-1 metric.
    Duration median 6.5 s (p5 2.8 s, p95 13.9 s, max 71 s), 25 / 29.97 fps, all 320x240.
        32 frames over the whole clip = stride ~5 frames (median), close to VideoMAE's
        pre-training stride of 4 -> keep the NSVA sampling (32 frames -> 16 tokens).
    Audio: 6,837 clips (51%) have an mp3 track, 6,738 audible; 37 classes always have audio,
        50 classes never, 14 mixed. Audio presence is almost a class-level property.
        -> silent/no-audio clips are masked (has_audio from RMS), modality dropout on audio
           for v3/v4, and an extra evaluation on the 51 mostly-audible classes ("audio51").
    Training split has 18 groups -> hold out 3 whole groups (~1.6k clips) for model selection,
        then refit on the full training list for the selected number of epochs.
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
PRETRAINED = os.environ.get("PB_PRETRAINED", os.path.join(os.path.dirname(HERE), "common", "pretrained"))
DATA_ROOT = "/data/Cuong/pickleball/data/ucf101"
VIDEO_ROOT = f"{DATA_ROOT}/UCF-101"
SPLIT_DIR = f"{DATA_ROOT}/ucfTrainTestlist"
ANALYSIS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis", "ucf101_videos.csv")
FEATURES = os.path.join(HERE, "cache")             # frozen features (extract_features.py)
OUTPUT_ROOT = os.path.join(HERE, "outputs", "runs")  # frozen/<version>, ft_k400_av/split<k>

# Self-supervised K400 checkpoint (no Kinetics labels): comparable with VideoMAE's
# 96.1 / 73.3 UCF/HMDB transfer numbers. The "finetuned-kinetics" checkpoint would leak labels.
VIDEOMAE = f"{PRETRAINED}/videomae-base"  # = MCG-NJU/videomae-base encoder
BEATS_CHECKPOINT = f"{PRETRAINED}/BEATs_iter3_plus_AS2M.pt"
BEATS_FT_CHECKPOINT = f"{PRETRAINED}/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"
BEATS_ROOT = f"{PRETRAINED}/unilm"
CLIP = "openai/clip-vit-base-patch32"
AUDIO_TOKENS = 32
FLIP_VIEW = True               # extra horizontally-flipped VideoMAE view (train augmentation)

SEED = 42
SPLITS = (1, 2, 3)
VAL_GROUPS = 3                 # of the 18 training groups
REFIT = True

# NSVA used batch 256 on ~3k clips; ~8k clips here, batch 128 gives ~63 steps/epoch.
BATCH_SIZE = 128
EPOCHS = 60
WARMUP_EPOCHS = 5
PATIENCE = 12
LR = 3e-4
WEIGHT_DECAY = 0.05            # 0.01 on NSVA; 101 classes x ~95 clips overfits faster
GRAD_CLIP = 1.0
LABEL_SMOOTHING = 0.1

# Model (same backbone-free heads as NSVA; head widened for 101 classes)
FUSION_DIM = 768
CROSS_HEADS = 16
TEMPORAL_HEADS = 12
TEMPORAL_LAYERS = 2
DROPOUT = 0.1
HEAD_HIDDEN = 512              # 256 on NSVA (3 classes); a 256 bottleneck is tight for 101
HEAD_DROPOUT = 0.3

# Feature-level augmentation
FLIP_PROB = 0.5
TOKEN_DROP = 0.1               # zero 10% of the 16 temporal tokens per training clip
AUDIO_DROPOUT = 0.3            # v3/v4: drop the audio of 30% of training clips
