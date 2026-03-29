from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .filter import ConstantVelocityKalmanFilter
from .measurements import CameraMeasurement, LidarMeasurement
from .track import Track


CHI2_95_THRESHOLDS = {
    1: 3.841458820694124,
    2: 5.991464547107979,
    3: 7.814727903251179,
}


@dataclass(frozen=True)
class AssociationResult:
    matches: list[tuple[int, int, float]]
    unmatched_track_indices: list[int]
    unmatched_measurement_indices: list[int]
    association_matrix: np.ndarray


class NearestNeighborAssociation:
    """Greedy nearest-neighbor association with Mahalanobis gating."""

    def __init__(self, gating_threshold: float | None = None) -> None:
        self.gating_threshold = float(gating_threshold) if gating_threshold is not None else None
        self.association_matrix = np.empty((0, 0), dtype=float)

    def _gate_threshold(self, measurement_dim: int) -> float:
        if self.gating_threshold is not None:
            return self.gating_threshold
        return CHI2_95_THRESHOLDS[measurement_dim]

    def mahalanobis_distance(
        self,
        track: Track,
        measurement: LidarMeasurement | CameraMeasurement,
        kalman_filter: ConstantVelocityKalmanFilter,
        sensor_model=None,
    ) -> float:
        measurement_dim = measurement.z.shape[0]
        if sensor_model is None:
            z_pred, S = kalman_filter.project(track.x, track.P, measurement.R)
        else:
            z_pred, S, _ = kalman_filter.project_extended(track.x, track.P, measurement.R, sensor_model)
        gamma = measurement.z.reshape(measurement_dim, 1) - z_pred.reshape(measurement_dim, 1)
        distance = float((gamma.transpose() @ np.linalg.inv(S) @ gamma).item())
        return distance

    def is_within_gate(self, distance: float, measurement_dim: int) -> bool:
        return float(distance) <= self._gate_threshold(measurement_dim)

    def associate(
        self,
        tracks: list[Track],
        measurements: list[LidarMeasurement] | list[CameraMeasurement],
        kalman_filter: ConstantVelocityKalmanFilter,
        sensor_model=None,
    ) -> AssociationResult:
        num_tracks = len(tracks)
        num_measurements = len(measurements)

        if num_tracks == 0 or num_measurements == 0:
            self.association_matrix = np.full((num_tracks, num_measurements), np.inf, dtype=float)
            return AssociationResult(
                matches=[],
                unmatched_track_indices=list(range(num_tracks)),
                unmatched_measurement_indices=list(range(num_measurements)),
                association_matrix=self.association_matrix.copy(),
            )

        matrix = np.full((num_tracks, num_measurements), np.inf, dtype=float)
        candidates: list[tuple[float, int, int]] = []

        for track_idx, track in enumerate(tracks):
            for meas_idx, measurement in enumerate(measurements):
                distance = self.mahalanobis_distance(
                    track,
                    measurement,
                    kalman_filter,
                    sensor_model=sensor_model,
                )
                if self.is_within_gate(distance, measurement.z.shape[0]):
                    matrix[track_idx, meas_idx] = distance
                    candidates.append((distance, track_idx, meas_idx))

        candidates.sort(key=lambda item: item[0])

        matched_track_indices: set[int] = set()
        matched_measurement_indices: set[int] = set()
        matches: list[tuple[int, int, float]] = []

        for distance, track_idx, meas_idx in candidates:
            if track_idx in matched_track_indices or meas_idx in matched_measurement_indices:
                continue
            matched_track_indices.add(track_idx)
            matched_measurement_indices.add(meas_idx)
            matches.append((track_idx, meas_idx, distance))

        self.association_matrix = matrix
        return AssociationResult(
            matches=matches,
            unmatched_track_indices=[
                idx for idx in range(num_tracks) if idx not in matched_track_indices
            ],
            unmatched_measurement_indices=[
                idx for idx in range(num_measurements) if idx not in matched_measurement_indices
            ],
            association_matrix=matrix.copy(),
        )
