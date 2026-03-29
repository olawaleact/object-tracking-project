from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .measurements import LidarMeasurement


@dataclass
class Track:
    """Tracked object with CV filter state and box metadata for plotting/reporting."""

    id: int
    x: np.ndarray
    P: np.ndarray
    state: str
    l: float
    w: float
    h: float
    yaw: float
    label: str
    score: float
    hits: int = 1
    age: int = 1
    misses: int = 0
    time_since_update: int = 0

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=float).reshape(6)
        self.P = np.asarray(self.P, dtype=float).reshape(6, 6)

    @classmethod
    def from_measurement(
        cls,
        measurement: LidarMeasurement,
        track_id: int,
        velocity_std: tuple[float, float, float] = (10.0, 10.0, 2.0),
    ) -> "Track":
        x = np.zeros(6, dtype=float)
        x[0:3] = measurement.z

        P = np.zeros((6, 6), dtype=float)
        P[0:3, 0:3] = measurement.R
        P[3:6, 3:6] = np.diag(np.square(np.asarray(velocity_std, dtype=float).reshape(3)))

        return cls(
            id=int(track_id),
            x=x,
            P=P,
            state="initialized",
            l=measurement.l,
            w=measurement.w,
            h=measurement.h,
            yaw=measurement.yaw,
            label=measurement.label,
            score=measurement.score,
        )

    def set_prediction(self, x: np.ndarray, P: np.ndarray) -> None:
        self.x = np.asarray(x, dtype=float).reshape(6)
        self.P = np.asarray(P, dtype=float).reshape(6, 6)
        self.age += 1
        self.time_since_update += 1

    def apply_measurement_update(
        self,
        x: np.ndarray,
        P: np.ndarray,
        measurement: LidarMeasurement,
        confirmation_hits: int,
    ) -> None:
        self.x = np.asarray(x, dtype=float).reshape(6)
        self.P = np.asarray(P, dtype=float).reshape(6, 6)
        self.hits += 1
        self.misses = 0
        self.time_since_update = 0
        self.update_metadata(measurement)
        self.refresh_state(confirmation_hits=confirmation_hits)

    def apply_camera_update(
        self,
        x: np.ndarray,
        P: np.ndarray,
        confirmation_hits: int,
    ) -> None:
        """Apply a camera-only EKF update without changing 3D box metadata."""

        self.x = np.asarray(x, dtype=float).reshape(6)
        self.P = np.asarray(P, dtype=float).reshape(6, 6)
        self.hits += 1
        self.misses = 0
        self.time_since_update = 0
        self.refresh_state(confirmation_hits=confirmation_hits)

    def update_metadata(self, measurement: LidarMeasurement) -> None:
        self.l = measurement.l
        self.w = measurement.w
        self.h = measurement.h
        self.yaw = measurement.yaw
        self.label = measurement.label
        self.score = measurement.score

    def refresh_state(self, confirmation_hits: int = 3) -> None:
        if self.hits >= confirmation_hits:
            self.state = "confirmed"
        elif self.hits >= 2:
            self.state = "tentative"
        else:
            self.state = "initialized"

    def mark_missed(self) -> None:
        self.misses += 1

    def __repr__(self) -> str:
        pos = np.round(self.x[0:3], 2).tolist()
        vel = np.round(self.x[3:6], 2).tolist()
        return (
            f"Track(id={self.id}, state='{self.state}', label='{self.label}', "
            f"pos={pos}, vel={vel}, score={self.score:.2f}, hits={self.hits}, misses={self.misses})"
        )
