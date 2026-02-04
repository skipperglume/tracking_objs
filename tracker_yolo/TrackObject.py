import numpy as np
from dataclasses import dataclass
from tracker_yolo.utils import create_tracker, bbox_center
from tracker_yolo.KalmanFilter2D import KalmanFilter2D


@dataclass
class Detection:
    """
    Class for objects to hold detection results.
    """

    bbox: np.ndarray  # [x1, y1, x2, y2] - Bounding box on a frame
    conf: float  # Confidence score
    cls: int  # Class


class TrackedObject:
    _next_id = 0

    def __init__(self, detection: Detection, fps: float, frame):
        self.id = TrackedObject._next_id
        TrackedObject._next_id += 1

        x1, y1, x2, y2 = detection.bbox
        self.tracker = create_tracker()
        self.tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))

        cx, cy = bbox_center(detection.bbox)
        self.kf = KalmanFilter2D(cx, cy, dt=1 / fps)  # adjust FPS

        self.bbox = detection.bbox
        self.cls = detection.cls
        self.conf = detection.conf

        # Variables to store history (For velocity estimation, etc.)
        self.fame_ids = []  # List of indices of frames where this object was detected
        self.timestamps = []  # List of timestamps corresponding to the frames
        self.centers = []  # List of center points of the bounding boxes

        self.hits = 1  # number of successful matches when associating new detections to existing tracks
        self.age = 0  # total frames alive
        self.missed = 0  # frames since last match

    @classmethod
    def reset_ids(cls):
        cls._next_id = 0

    def predict(self):
        """
        For now: no motion model.
        Later: Kalman filter lives here.
        """
        return self.bbox

    def _update_bbox_from_center(self, cx, cy):
        """
        Update bbox from center coordinates.
        """
        x1, y1, x2, y2 = self.bbox
        w = x2 - x1
        h = y2 - y1
        self.bbox = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    def predict_kf(self):
        """
        Predict the next position using Kalman Filter.
        """
        cx, cy = self.kf.predict()
        self._update_bbox_from_center(cx, cy)
        self.missed += 1

    def update_timeticks(self, frame_id: int, fps: float):
        timestamp = frame_id / fps
        self.fame_ids.append(frame_id)
        self.centers.append(bbox_center(self.bbox))
        self.timestamps.append(timestamp)

    def update(self, detection: Detection):
        self.bbox = detection.bbox
        self.conf = detection.conf
        self.missed = 0
        self.age += 1
        self.hits += 1

        cx, cy = bbox_center(detection.bbox)
        self.kf.update(cx, cy)

    def update_from_tracker(self, frame):
        ok, box = self.tracker.update(frame)
        if not ok:
            self.missed += 1
            return False

        x, y, w, h = box
        self.bbox = np.array([x, y, x + w, y + h])
        self.missed = 0
        self.age += 1
        cx, cy = bbox_center(self.bbox)
        self.kf.update(cx, cy)

        return True

    def mark_missed(self):
        """
        Mark this object as missed in the current frame.
        """
        self.missed += 1
        self.age += 1
