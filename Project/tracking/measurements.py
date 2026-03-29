from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from detection.types import Detection3D
from tools.frame_pb2 import Camera


OBJECT_TYPE_TO_LABEL = {
    0: "car",
    1: "truck",
    2: "pedestrian",
    3: "cyclist",
}


@dataclass(frozen=True)
class LidarMeasurement:
    """Minimal LiDAR position measurement built from a project-side Detection3D."""

    z: np.ndarray
    R: np.ndarray
    l: float
    w: float
    h: float
    yaw: float
    label: str
    score: float

    def __post_init__(self) -> None:
        z = np.asarray(self.z, dtype=float).reshape(3)
        R = np.asarray(self.R, dtype=float).reshape(3, 3)
        object.__setattr__(self, "z", z)
        object.__setattr__(self, "R", R)

    def as_column(self) -> np.ndarray:
        return self.z.reshape(3, 1)

    @classmethod
    def from_detection3d(
        cls,
        detection: Detection3D,
        position_std: tuple[float, float, float] = (1.0, 1.0, 0.5),
    ) -> "LidarMeasurement":
        std = np.asarray(position_std, dtype=float).reshape(3)
        covariance = np.diag(std**2)
        return cls(
            z=np.array([detection.x, detection.y, detection.z], dtype=float),
            R=covariance,
            l=float(detection.l),
            w=float(detection.w),
            h=float(detection.h),
            yaw=float(detection.yaw),
            label=detection.label,
            score=float(detection.score),
        )


@dataclass(frozen=True)
class CameraMeasurement:
    """Minimal camera image-space measurement using bbox center [u, v]."""

    z: np.ndarray
    R: np.ndarray
    width: float
    height: float
    label: str
    detection_id: str
    timestamp: str | None = None
    sensor_position: int | None = None

    def __post_init__(self) -> None:
        z = np.asarray(self.z, dtype=float).reshape(2)
        R = np.asarray(self.R, dtype=float).reshape(2, 2)
        object.__setattr__(self, "z", z)
        object.__setattr__(self, "R", R)

    def as_column(self) -> np.ndarray:
        return self.z.reshape(2, 1)

    @classmethod
    def from_camera_detection(
        cls,
        detection: Camera.CameraDetection,
        camera: Camera,
        pixel_std: tuple[float, float] = (5.0, 5.0),
    ) -> "CameraMeasurement":
        x0, y0, width, height = [float(value) for value in detection.bbox]
        std = np.asarray(pixel_std, dtype=float).reshape(2)
        covariance = np.diag(std**2)
        label = OBJECT_TYPE_TO_LABEL.get(int(detection.type), f"unknown_{int(detection.type)}")
        return cls(
            z=np.array([x0, y0], dtype=float),
            R=covariance,
            width=width,
            height=height,
            label=label,
            detection_id=str(detection.id),
            timestamp=str(camera.timestamp),
            sensor_position=int(camera.pos),
        )


def detection_to_lidar_measurement(
    detection: Detection3D,
    position_std: tuple[float, float, float] = (1.0, 1.0, 0.5),
) -> LidarMeasurement:
    return LidarMeasurement.from_detection3d(detection, position_std=position_std)


def detections_to_lidar_measurements(
    detections: Iterable[Detection3D],
    position_std: tuple[float, float, float] = (1.0, 1.0, 0.5),
) -> list[LidarMeasurement]:
    return [
        detection_to_lidar_measurement(detection, position_std=position_std)
        for detection in detections
    ]


def camera_detection_to_measurement(
    detection: Camera.CameraDetection,
    camera: Camera,
    pixel_std: tuple[float, float] = (5.0, 5.0),
) -> CameraMeasurement:
    return CameraMeasurement.from_camera_detection(
        detection,
        camera,
        pixel_std=pixel_std,
    )


def camera_detections_to_measurements(
    camera: Camera,
    pixel_std: tuple[float, float] = (5.0, 5.0),
) -> list[CameraMeasurement]:
    return [
        camera_detection_to_measurement(detection, camera, pixel_std=pixel_std)
        for detection in camera.detections
    ]
