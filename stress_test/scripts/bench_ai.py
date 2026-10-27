"""Đo thời gian từng giai đoạn của VideoInferenceService.predict (chạy trong container AI)."""
import sys, time, torch
from pickleball_track.core.config import get_settings
from pickleball_track.models.model_loader import load_models
from pickleball_track.services.inference_service import VideoInferenceService
import pickleball_track.services.inference_service as m

def main():
    s = get_settings()
    t = time.perf_counter(); bundle = load_models(s); print("load_models", round(time.perf_counter() - t, 1), "s", "batch", bundle.batch_size, flush=True)
    svc = VideoInferenceService(bundle, s.inference, slots=1)
    acc = {"model": 0.0, "batches": 0, "windows": 0}
    orig = svc.predict_batch
    def timed(batch, *a, **k):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        out = orig(batch, *a, **k)
        torch.cuda.synchronize(); acc["model"] += time.perf_counter() - t0; acc["batches"] += 1; acc["windows"] += len(batch.starts)
        return out
    svc.predict_batch = timed
    orig_audio = m.load_audio
    def ta(p):
        t0 = time.perf_counter(); r = orig_audio(p); acc["audio"] = time.perf_counter() - t0; return r
    m.load_audio = ta
    for p in sys.argv[1:]:
        for k in ("model", "batches", "windows"): acc[k] = 0
        t0 = time.perf_counter(); r = svc.predict(p); tot = time.perf_counter() - t0
        print(f"{p}: total {tot:.2f}s | audio {acc['audio']:.2f}s | model(GPU) {acc['model']:.2f}s over {acc['batches']} batches/{acc['windows']} windows "
              f"({acc['model']/max(acc['windows'],1)*1000:.1f} ms/window) | frames+other {tot-acc['model']-acc['audio']:.2f}s | hits {r['total_hits']} misses {r['total_misses']}", flush=True)
    svc.close(); bundle.frame_pool and bundle.frame_pool.close()

if __name__ == "__main__":
    main()
