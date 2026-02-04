import cv2
import numpy as np
from tracker_yolo.utils import bbox_center


def draw_dashed_line(
    img, p1, p2, color, thickness=2, dash_len=10, do_dashed: bool = True
):
    p1 = np.array(p1, dtype=np.float32)
    p2 = np.array(p2, dtype=np.float32)

    dist = np.linalg.norm(p2 - p1)
    if dist < 1:
        return

    direction = (p2 - p1) / dist
    num_dashes = int(dist // dash_len)
    if not do_dashed:
        num_dashes = 1
        dash_len = dist
    for i in range(0, num_dashes, 2):
        start = p1 + direction * dash_len * i
        end = p1 + direction * dash_len * (i + 1)
        cv2.line(
            img,
            tuple(start.astype(int)),
            tuple(end.astype(int)),
            color,
            thickness,
        )


def draw_kalman_prediction(frame, track, steps=10, color=(255, 255, 0)):
    points = track.kf.predict_n_steps(steps)

    prev = bbox_center(track.bbox)
    for p in points:
        draw_dashed_line(frame, prev, p, color, do_dashed=False)
        prev = p
