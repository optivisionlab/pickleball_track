# Runtime Model Weights

Keep the three model artifacts in this directory:

- `best.pt`: trained audiovisual fusion model (`{"model": state_dict, "label2id": ...}`).
- `BEATs_iter3_plus_AS2M.pt`: Microsoft BEATs pretrained audio model.
- `videomae-base/`: VideoMAE config, processor config, and `model.safetensors`.

The API loads these models once at startup and reuses them for every request in
the same worker process. Binary checkpoints are ignored by Git because they are
large deployment artifacts. Only load checkpoints from trusted sources.
