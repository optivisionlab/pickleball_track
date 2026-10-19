# Pickleball Track

| Folder | Content |
| --- | --- |
| [`score_video_block/`](score_video_block/README.md) | FastAPI service scoring pickleball videos (hit/miss detection), Dockerfile, tests |

Each block is self-contained: install, test and build Docker images from inside its folder, e.g.

```bash
cd score_video_block
python -m pip install -e ".[dev]"
python -m pytest -q
docker build -t pickleball-track .
```
