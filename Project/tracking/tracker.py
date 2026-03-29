from __future__ import annotations

from dataclasses import dataclass, field

from detection.types import Detection3D

from .association import AssociationResult, NearestNeighborAssociation
from .filter import ConstantVelocityKalmanFilter
from .measurements import (
    CameraMeasurement,
    LidarMeasurement,
    detections_to_lidar_measurements,
)
from .sensors import CameraSensorModel
from .track import Track


@dataclass
class MultiObjectTracker:
    """Minimal LiDAR-only multi-object tracker for the final notebook pipeline."""

    kalman_filter: ConstantVelocityKalmanFilter = field(default_factory=ConstantVelocityKalmanFilter)
    association: NearestNeighborAssociation = field(default_factory=NearestNeighborAssociation)
    confirmation_hits: int = 3
    max_missed: int = 2
    tracks: list[Track] = field(default_factory=list)
    next_track_id: int = 1

    def predict(self, dt: float = 1.0) -> None:
        for track in self.tracks:
            x_pred, P_pred = self.kalman_filter.predict(track.x, track.P, dt=dt)
            track.set_prediction(x_pred, P_pred)

    def update_lidar(self, measurements: list[LidarMeasurement]) -> AssociationResult:
        result = self.association.associate(self.tracks, measurements, self.kalman_filter)

        for track_idx, meas_idx, _distance in result.matches:
            track = self.tracks[track_idx]
            measurement = measurements[meas_idx]
            x_upd, P_upd = self.kalman_filter.update(track.x, track.P, measurement.z, measurement.R)
            track.apply_measurement_update(
                x=x_upd,
                P=P_upd,
                measurement=measurement,
                confirmation_hits=self.confirmation_hits,
            )

        for track_idx in result.unmatched_track_indices:
            self.tracks[track_idx].mark_missed()

        for meas_idx in result.unmatched_measurement_indices:
            self._start_new_track(measurements[meas_idx])

        self._delete_stale_tracks()
        return result

    def step(
        self,
        detections: list[Detection3D],
        dt: float = 1.0,
        position_std: tuple[float, float, float] = (1.0, 1.0, 0.5),
    ) -> list[Track]:
        measurements = detections_to_lidar_measurements(detections, position_std=position_std)
        self.predict(dt=dt)
        self.update_lidar(measurements)
        return list(self.tracks)

    def update_camera(
        self,
        measurements: list[CameraMeasurement],
        camera_sensor: CameraSensorModel,
    ) -> AssociationResult:
        """Update existing in-FOV tracks with camera image-center measurements.

        Assumptions:
        - track birth remains LiDAR-only
        - only tracks already inside the camera image are considered
        - unmatched camera measurements do not create new tracks
        """

        candidate_track_indices = [
            idx for idx, track in enumerate(self.tracks) if camera_sensor.in_fov(track.x)
        ]
        candidate_tracks = [self.tracks[idx] for idx in candidate_track_indices]

        result = self.association.associate(
            candidate_tracks,
            measurements,
            self.kalman_filter,
            sensor_model=camera_sensor,
        )

        full_matches: list[tuple[int, int, float]] = []
        for candidate_track_idx, meas_idx, distance in result.matches:
            full_track_idx = candidate_track_indices[candidate_track_idx]
            track = self.tracks[full_track_idx]
            measurement = measurements[meas_idx]
            x_upd, P_upd = self.kalman_filter.update_extended(
                track.x,
                track.P,
                measurement.z,
                measurement.R,
                camera_sensor,
            )
            track.apply_camera_update(
                x=x_upd,
                P=P_upd,
                confirmation_hits=self.confirmation_hits,
            )
            full_matches.append((full_track_idx, meas_idx, distance))

        return AssociationResult(
            matches=full_matches,
            unmatched_track_indices=[
                candidate_track_indices[idx] for idx in result.unmatched_track_indices
            ],
            unmatched_measurement_indices=result.unmatched_measurement_indices,
            association_matrix=result.association_matrix,
        )

    def _start_new_track(self, measurement: LidarMeasurement) -> None:
        track = Track.from_measurement(measurement, track_id=self.next_track_id)
        self.next_track_id += 1
        self.tracks.append(track)

    def _delete_stale_tracks(self) -> None:
        active_tracks: list[Track] = []
        for track in self.tracks:
            if track.state != "confirmed" and track.misses > 0:
                continue
            if track.state == "confirmed" and track.misses > self.max_missed:
                continue
            active_tracks.append(track)
        self.tracks = active_tracks
