import os
import glob
import cv2
import numpy as np

BALL_CLASS = 0
MAX_GAP = 5


# -------------------------------
# Kalman
# -------------------------------
def create_kalman():
    kf = cv2.KalmanFilter(4, 2)
    # state = [x,y,vx,vy]
    kf.transitionMatrix = np.array([
        [1,0,1,0],
        [0,1,0,1],
        [0,0,1,0],
        [0,0,0,1]
    ], np.float32)

    kf.measurementMatrix = np.array([
        [1,0,0,0],
        [0,1,0,0]
    ], np.float32)

    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-3
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-2
    kf.errorCovPost = np.eye(4, dtype=np.float32)
    return kf


# -------------------------------
# Read one label
# -------------------------------
def read_label(path):

    others = []
    ball = None

    if not os.path.exists(path):
        return ball, others
    with open(path) as f:
        lines = f.readlines()
    for line in lines:
        if len(line.strip()) == 0:
            continue
        cls, x, y, w, h = line.split()
        cls = int(cls)
        if cls == BALL_CLASS:
            ball = (
                float(x),
                float(y),
                float(w),
                float(h)
            )
        else:
            others.append(line.strip())
    return ball, others


# -------------------------------
# Write label
# -------------------------------
def write_label(path, ball, others):

    with open(path, "w") as f:

        if ball is not None:
            x, y, w, h = ball
            f.write(
                f"{BALL_CLASS} "
                f"{x:.6f} {y:.6f} "
                f"{w:.6f} {h:.6f}\n"
            )

        for line in others:
            f.write(line + "\n")

def process_label_folder(label_dir):
    save_dir = os.path.join(
        os.path.dirname(label_dir),
        "labels_paddles_tracked"
    )
    os.makedirs(save_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(label_dir, "*.txt")))

    if len(files) == 0:
        return

    trajectory = []
    others_all = []

    for f in files:
        ball, others = read_label(f)
        trajectory.append(ball)
        others_all.append(others)

    # -------------------------------
    # Kalman prediction
    # -------------------------------
    kf = create_kalman()

    predictions = [None] * len(files)
    initialized = False

    for i, ball in enumerate(trajectory):

        if ball is not None:
            x, y, w, h = ball
            measure = np.array([[x], [y]], np.float32)
            if not initialized:
                kf.statePost = np.array(
                    [[x], [y], [0], [0]],
                    np.float32
                )
                initialized = True
            else:
                kf.predict()
                kf.correct(measure)

        else:
            if initialized:
                pred = kf.predict()
                predictions[i] = (
                    float(pred[0, 0]),
                    float(pred[1, 0])
                )

    # -------------------------------
    # Fill gaps
    # -------------------------------
    result = trajectory.copy()

    i = 0

    while i < len(files):

        if trajectory[i] is not None:
            i += 1
            continue
        start = i
        while i < len(files) and trajectory[i] is None:
            i += 1
        end = i
        gap = end - start
        if gap > MAX_GAP:
            continue
        if start == 0 or end >= len(files):
            continue
        prev_ball = trajectory[start - 1]
        next_ball = trajectory[end]
        if prev_ball is None or next_ball is None:
            continue
        x1, y1, w1, h1 = prev_ball
        x2, y2, w2, h2 = next_ball
        for k in range(gap):
            alpha = (k + 1) / (gap + 1)
            xi = x1 + alpha * (x2 - x1)
            yi = y1 + alpha * (y2 - y1)
            wi = w1 + alpha * (w2 - w1)
            hi = h1 + alpha * (h2 - h1)
            if predictions[start + k] is not None:
                px, py = predictions[start + k]
                xi = 0.90 * xi + 0.10 * px
                yi = 0.90 * yi + 0.10 * py
            result[start + k] = (xi, yi, wi, hi)

    # -------------------------------
    # Save
    # -------------------------------
    for file, original_ball, new_ball, others in zip(
        files, trajectory, result, others_all
    ):
        save_path = os.path.join(save_dir, os.path.basename(file))
        write_label(save_path, new_ball, others)
        print(f"Done: {label_dir}")
    
# -------------------------------
# MAIN
# -------------------------------
root_dir = "/data/Cuong/pickleball/data/shot1"

label_dirs = []

for dirpath, dirnames, filenames in os.walk(root_dir):
    if os.path.basename(dirpath) == "labels_paddles":
        label_dirs.append(dirpath)

print(f"Found {len(label_dirs)}")

for label_dir in sorted(label_dirs):
    process_label_folder(label_dir)

print("All Done.")

