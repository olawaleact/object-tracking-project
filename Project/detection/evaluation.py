from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from detection.types import Detection3D
from tools.frame_pb2 import Frame


OBJECT_TYPE_TO_LABEL = {
    0: "car",
    1: "truck",
    2: "pedestrian",
    3: "cyclist",
}


@dataclass(frozen=True)
class DetectionMatch:
    """A single prediction-to-ground-truth match for one frame."""

    prediction: Detection3D
    ground_truth: Detection3D
    distance_xy: float


@dataclass
class FrameDetectionEval:
    """Simple one-frame evaluation result based on greedy center-distance matching."""

    predictions: list[Detection3D]
    ground_truth: list[Detection3D]
    matches: list[DetectionMatch]
    unmatched_predictions: list[Detection3D]
    unmatched_ground_truth: list[Detection3D]

    @property
    def num_predictions(self) -> int:
        return len(self.predictions)

    @property
    def num_ground_truth(self) -> int:
        return len(self.ground_truth)

    @property
    def num_matched(self) -> int:
        return len(self.matches)


def _normalize_allowed_labels(allowed_labels: Iterable[str] | None) -> set[str] | None:
    if allowed_labels is None:
        return None
    return {label.lower() for label in allowed_labels}


def lidar_gt_to_detection3d(detection) -> Detection3D:
    """Convert one protobuf LiDARDetection into a project-side Detection3D.

    Dataset LiDAR ground truth stores:
    - pos: center x, y, z
    - scale: size x, y, z
    - rot: roll, pitch, yaw
    - type: ObjectType enum
    """

    label = OBJECT_TYPE_TO_LABEL.get(int(detection.type), f"unknown_{int(detection.type)}")
    return Detection3D(
        x=float(detection.pos[0]),
        y=float(detection.pos[1]),
        z=float(detection.pos[2]),
        l=float(detection.scale[0]),
        w=float(detection.scale[1]),
        h=float(detection.scale[2]),
        yaw=float(detection.rot[2]),
        score=1.0,
        label=label,
    )


def extract_gt_lidar_detections(
    frame: Frame,
    lidar_index: int = 0,
    allowed_labels: Iterable[str] | None = None,
) -> list[Detection3D]:
    """Extract project-side GT detections from one LiDAR in a protobuf frame."""

    allowed = _normalize_allowed_labels(allowed_labels)
    detections: list[Detection3D] = []

    for detection in frame.lidars[lidar_index].detections:
        gt = lidar_gt_to_detection3d(detection)
        if allowed is None or gt.label in allowed:
            detections.append(gt)

    return detections


def evaluate_frame_detections(
    predictions: list[Detection3D],
    ground_truth: list[Detection3D],
    max_center_distance: float = 2.0,
    require_label_match: bool = True,
) -> FrameDetectionEval:
    """Greedily match predictions to GT using 2D center distance in the LiDAR plane.

    This is intentionally simple for first-pass evaluation on one frame.
    """

    candidates: list[tuple[float, int, int]] = []
    for pred_idx, pred in enumerate(predictions):
        for gt_idx, gt in enumerate(ground_truth):
            if require_label_match and pred.label != gt.label:
                continue
            distance_xy = float(np.hypot(pred.x - gt.x, pred.y - gt.y))
            if distance_xy <= max_center_distance:
                candidates.append((distance_xy, pred_idx, gt_idx))

    candidates.sort(key=lambda item: item[0])

    used_predictions: set[int] = set()
    used_ground_truth: set[int] = set()
    matches: list[DetectionMatch] = []

    for distance_xy, pred_idx, gt_idx in candidates:
        if pred_idx in used_predictions or gt_idx in used_ground_truth:
            continue
        used_predictions.add(pred_idx)
        used_ground_truth.add(gt_idx)
        matches.append(
            DetectionMatch(
                prediction=predictions[pred_idx],
                ground_truth=ground_truth[gt_idx],
                distance_xy=distance_xy,
            )
        )

    unmatched_predictions = [pred for idx, pred in enumerate(predictions) if idx not in used_predictions]
    unmatched_ground_truth = [gt for idx, gt in enumerate(ground_truth) if idx not in used_ground_truth]

    return FrameDetectionEval(
        predictions=predictions,
        ground_truth=ground_truth,
        matches=matches,
        unmatched_predictions=unmatched_predictions,
        unmatched_ground_truth=unmatched_ground_truth,
    )
