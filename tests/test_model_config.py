from pathlib import Path

from pickleball_track.core.config import (
    BEATS_CHECKPOINT,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_VIDEOMAE,
    WEIGHTS_DIR,
)


def test_default_models_use_the_single_runtime_weights_directory():
    assert WEIGHTS_DIR == Path(__file__).resolve().parents[1] / "src" / "weights"
    assert DEFAULT_CHECKPOINT_PATH == WEIGHTS_DIR / "best.pt"
    assert Path(BEATS_CHECKPOINT) == WEIGHTS_DIR / "BEATs_iter3_plus_AS2M.pt"
    assert Path(DEFAULT_VIDEOMAE) == WEIGHTS_DIR / "videomae-base"