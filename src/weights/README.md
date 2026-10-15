# Runtime Model Weights

Keep the three model artifacts in this directory:

- `best.pt`: trained audiovisual fusion model.
- `BEATs_iter3_plus_AS2M.pt`: Microsoft BEATs pretrained audio model.
- `videomae-base/`: VideoMAE config, processor config, and `model.safetensors`.

The API loads these models lazily on the first video request and reuses them
for later requests in the same worker process. Binary checkpoints are ignored
by Git because they are large deployment artifacts.