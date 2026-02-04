import cv2
import numpy as np


def create_tracker():
    return cv2.legacy.TrackerCSRT_create()


def create_tracker_specified(tracker_type: str):
    tracker_types = {
        "BOOSTING": cv2.TrackerBoosting_create,
        "MIL": cv2.TrackerMIL_create,
        "KCF": cv2.TrackerKCF_create,
        "TLD": cv2.TrackerTLD_create,
        "MEDIANFLOW": cv2.TrackerMedianFlow_create,
        "GOTURN": cv2.TrackerGOTURN_create,
        "MOSSE": cv2.TrackerMOSSE_create,
        "CSRT": cv2.TrackerCSRT_create,
    }
    if tracker_type in tracker_types:
        return tracker_types[tracker_type]()
    else:
        raise ValueError(f"Unknown tracker type: {tracker_type}")


def bbox_center(box):
    """
    Take a plain average to get the center of a bounding box.
    """

    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])


def iou(boxA, boxB):
    """
    Method to compute Intersection over Union (IoU) between two bounding boxes.
    Compute area of overlap / area of union
    If intersection is zero, returns 0
    """

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)  # area of intersection
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])  # area of boxA
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])  # area of boxB

    union = areaA + areaB - inter  # area of union
    return inter / union if union > 0 else 0  # Return IoU value


def center_distance(boxA, boxB):
    """
    Evaluate the Euclidean distance between the centers of two bounding boxes.
    """
    return np.linalg.norm(bbox_center(boxA) - bbox_center(boxB))


def associate_detections(tracked_objects, detections, iou_threshold=0.3):
    """
    Method that
    """

    matches = []  # list of (track_idx, detection_idx) tuples
    unmatched_tracks = set(
        range(len(tracked_objects))
    )  # set of unmatched track indices
    unmatched_dets = set(range(len(detections)))  # set of unmatched detection indices

    for t_idx, track in enumerate(tracked_objects):
        best_iou = 0
        best_d_idx = None  # Best detection index

        for d_idx in unmatched_dets:
            score = iou(track.predict(), detections[d_idx].bbox)
            if score > best_iou:
                best_iou = score
                best_d_idx = d_idx

        if best_iou > iou_threshold:
            matches.append((t_idx, best_d_idx))
            unmatched_tracks.remove(t_idx)
            unmatched_dets.remove(best_d_idx)

    return matches, unmatched_tracks, unmatched_dets


def overlaps_existing_track(detection, tracked_objects, iou_threshold=0.75):
    for track in tracked_objects:
        if iou(track.bbox, detection.bbox) > iou_threshold:
            return True
    return False


def bbox_length_px(bbox):
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    return max(w, h)


def estimate_speed(track, car_length_m=4.3):
    """
    Estimate speed of the tracked object in meters per second using the change in position in the last two frames.
    """

    if len(track.centers) < 2:
        return None

    position_1 = track.centers[-2]
    position_2 = track.centers[-1]
    dt = track.timestamps[-1] - track.timestamps[-2]

    dposition_px = np.linalg.norm(position_2 - position_1)

    L_px = bbox_length_px(track.bbox)
    meters_per_pixel = car_length_m / L_px

    speed_mps = dposition_px * meters_per_pixel / dt
    return speed_mps


def estimate_speed_mean(track, car_length_m=4.3):
    """
    Estimate speed of the tracked object in meters per second using the average change in position over all frames.
    """

    if len(track.centers) < 2:
        return None

    position_diff = np.diff(np.array(track.centers), axis=0)
    distances_px = np.linalg.norm(position_diff, axis=1)
    L_px = bbox_length_px(track.bbox)
    meters_per_pixel = car_length_m / L_px
    total_distance_m = np.sum(distances_px) * meters_per_pixel
    total_time_s = track.timestamps[-1] - track.timestamps[0]
    speed_mps = total_distance_m / total_time_s
    return speed_mps


def smooth_speed(prev, new, alpha=0.3):
    if prev is None:
        return new
    return alpha * new + (1 - alpha) * prev
