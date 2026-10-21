from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

from pickleball_track.pipelines.frames import ProcessFramePool
from pickleball_track.services.inference_service import VideoInferenceService


def test_slots_run_videos_concurrently_with_identical_results(tiny_models, clip_path):
    expected = VideoInferenceService(tiny_models).predict(clip_path)
    service = VideoInferenceService(tiny_models, slots=2)

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda _: service.predict(clip_path), range(3)))

    assert results == [expected] * 3
    assert service.busy_slots == 0


def test_each_slot_gets_its_own_preprocessing_pool(tiny_models, clip_path):
    pool = ProcessFramePool(tiny_models.processor, workers=1)
    models = replace(tiny_models, frame_pool=pool)
    service = VideoInferenceService(models, slots=2)
    try:
        assert len(service._extra_pools) == 1 and service._extra_pools[0] is not pool
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda _: service.predict(clip_path), range(2)))
        assert results[0] == results[1] == VideoInferenceService(tiny_models).predict(clip_path)
    finally:
        service.close()
        pool.close()
