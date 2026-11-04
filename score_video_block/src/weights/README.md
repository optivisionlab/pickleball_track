# Runtime Model Weights

Keep the three model artifacts in this directory:

- `pickleball_v4.pt`: the trained v4 model, one network per cross-validation fold
  (averaged at inference) with each fold's feature mean/std and the CLIP
  embeddings of the class descriptions. Build it from a training run with
  `scripts/export_checkpoint.py`.
- `BEATs_iter3_plus_AS2M.pt`: Microsoft BEATs pretrained audio model.
- `videomae-base-finetuned-kinetics/`: VideoMAE config and `model.safetensors`.
  The weights are downloaded from `MCG-NJU/videomae-base-finetuned-kinetics` at
  the first startup if absent (`PICKLEBALL_VIDEOMAE_REPO`; the directory must be
  writable, also when mounted into Docker). The committed config is kept as it is.

The BEATs source code is not a weight: clone it into `vendor/unilm` (see the
project README).

The API loads these models once at startup and reuses them for every request in
the same worker process. Binary checkpoints are ignored by Git because they are
large deployment artifacts. Only load checkpoints from trusted sources.
