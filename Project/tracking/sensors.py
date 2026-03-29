from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tools.frame_pb2 import Camera


def _extract_position(state: np.ndarray) -> np.ndarray:
    array = np.asarray(state, dtype=float).reshape(-1)
    if array.shape[0] < 3:
        raise ValueError("CameraSensorModel expects a 3D point or a state with at least 3 position entries.")
    return array[:3]


@dataclass(frozen=True)
class CameraSensorModel:
    """Minimal pinhole camera model for projecting vehicle-frame 3D points to image center [u, v].

    Assumptions:
    - input state is in vehicle coordinates
    - only the first three state elements [x, y, z] are used
    - distortion coefficients are stored but ignored in this first geometry layer
    - projection uses the raw camera intrinsics and vehicle-to-camera extrinsics
    """

    T: np.ndarray
    K: np.ndarray
    D: np.ndarray
    width: int
    height: int
    timestamp: str | None = None
    position: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "T", np.asarray(self.T, dtype=float).reshape(4, 4))
        object.__setattr__(self, "K", np.asarray(self.K, dtype=float).reshape(3, 3))
        object.__setattr__(self, "D", np.asarray(self.D, dtype=float).reshape(-1))
        object.__setattr__(self, "width", int(self.width))
        object.__setattr__(self, "height", int(self.height))

    @classmethod
    def from_camera(cls, camera: Camera) -> "CameraSensorModel":
        return cls(
            T=np.asarray(camera.T, dtype=float).reshape(4, 4),
            K=np.asarray(camera.K, dtype=float).reshape(3, 3),
            D=np.asarray(camera.D, dtype=float),
            width=int(camera.width),
            height=int(camera.height),
            timestamp=str(camera.timestamp),
            position=int(camera.pos),
        )

    @property
    def R_cam_veh(self) -> np.ndarray:
        return self.T[:3, :3]

    @property
    def t_cam_veh(self) -> np.ndarray:
        return self.T[:3, 3]

    def vehicle_to_camera(self, state: np.ndarray) -> np.ndarray:
        position = _extract_position(state)
        homogeneous = np.ones(4, dtype=float)
        homogeneous[:3] = position
        camera_point = self.T @ homogeneous
        return camera_point[:3]

    def project_vehicle_point(self, state: np.ndarray) -> tuple[np.ndarray, float]:
        camera_point = self.vehicle_to_camera(state)
        projected = self.K @ camera_point
        depth = float(projected[2])
        if depth <= 0:
            raise ValueError("Point projects behind the camera or onto the camera plane.")
        uv = projected[:2] / depth
        return uv.astype(float), depth

    def hx(self, state: np.ndarray) -> np.ndarray:
        uv, _depth = self.project_vehicle_point(state)
        return uv

    def calculate_jacobian(self, state: np.ndarray) -> np.ndarray:
        position = _extract_position(state)
        camera_point = self.vehicle_to_camera(position)

        row0 = self.K[0, :]
        row1 = self.K[1, :]
        row2 = self.K[2, :]

        numerator_u = float(row0 @ camera_point)
        numerator_v = float(row1 @ camera_point)
        denominator = float(row2 @ camera_point)
        if denominator <= 0:
            raise ValueError("Point projects behind the camera or onto the camera plane.")

        d_num_u = row0 @ self.R_cam_veh
        d_num_v = row1 @ self.R_cam_veh
        d_den = row2 @ self.R_cam_veh

        du_dpos = (d_num_u * denominator - numerator_u * d_den) / (denominator**2)
        dv_dpos = (d_num_v * denominator - numerator_v * d_den) / (denominator**2)

        H = np.zeros((2, 6), dtype=float)
        H[0, :3] = du_dpos
        H[1, :3] = dv_dpos
        return H

    def in_fov(self, state: np.ndarray) -> bool:
        try:
            uv, _depth = self.project_vehicle_point(state)
        except ValueError:
            return False

        u, v = float(uv[0]), float(uv[1])
        return 0.0 <= u < float(self.width) and 0.0 <= v < float(self.height)
