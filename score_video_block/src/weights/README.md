# Runtime Model Weights

Keep the three model artifacts in this directory:

- `best.pt`: trained audiovisual fusion model (`{"model": state_dict, "label2id": ...}`).
- `BEATs_iter3_plus_AS2M.pt`: Microsoft BEATs pretrained audio model.
- `videomae-base/`: VideoMAE config, processor config, and `model.safetensors`.
  The weights are downloaded from `MCG-NJU/videomae-base` at the first startup
  if absent (`PICKLEBALL_VIDEOMAE_REPO`; the directory must be writable, also
  when mounted into Docker). The committed configs are kept as they are.

The BEATs source code is not a weight: clone it into `vendor/unilm` (see the
project README).

The API loads these models once at startup and reuses them for every request in
the same worker process. Binary checkpoints are ignored by Git because they are
large deployment artifacts. Only load checkpoints from trusted sources.
