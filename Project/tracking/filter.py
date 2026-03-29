from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _as_column(vector: np.ndarray | list[float] | tuple[float, ...], dim: int) -> np.ndarray:
    array = np.asarray(vector, dtype=float).reshape(-1)
    if array.shape[0] != dim:
        raise ValueError(f"Expected vector with {dim} elements, got shape {array.shape}.")
    return array.reshape(dim, 1)


@dataclass
class ConstantVelocityKalmanFilter:
    """Simple constant-velocity Kalman filter for state [x, y, z, vx, vy, vz]."""

    process_var: float = 1.0

    @property
    def dim_state(self) -> int:
        return 6

    @property
    def dim_meas(self) -> int:
        return 3

    def transition_matrix(self, dt: float) -> np.ndarray:
        F = np.eye(self.dim_state, dtype=float)
        F[0:3, 3:6] = np.eye(3, dtype=float) * float(dt)
        return F

    def process_noise(self, dt: float) -> np.ndarray:
        dt = float(dt)
        q = float(self.process_var)
        q_pos = (dt**4) / 4.0 * q
        q_cross = (dt**3) / 2.0 * q
        q_vel = (dt**2) * q

        Q = np.zeros((self.dim_state, self.dim_state), dtype=float)
        eye3 = np.eye(3, dtype=float)
        Q[0:3, 0:3] = eye3 * q_pos
        Q[0:3, 3:6] = eye3 * q_cross
        Q[3:6, 0:3] = eye3 * q_cross
        Q[3:6, 3:6] = eye3 * q_vel
        return Q

    def measurement_matrix(self) -> np.ndarray:
        H = np.zeros((self.dim_meas, self.dim_state), dtype=float)
        H[0:3, 0:3] = np.eye(3, dtype=float)
        return H

    def predict(self, x: np.ndarray, P: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Predict the next state and covariance using the exercise notebook KF equations."""

        x_col = _as_column(x, self.dim_state)
        P = np.asarray(P, dtype=float).reshape(self.dim_state, self.dim_state)

        F = self.transition_matrix(dt)
        Q = self.process_noise(dt)

        x_pred = F @ x_col
        P_pred = F @ P @ F.transpose() + Q
        return x_pred[:, 0], P_pred

    def project(self, x: np.ndarray, P: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project state uncertainty into measurement space."""

        x_col = _as_column(x, self.dim_state)
        P = np.asarray(P, dtype=float).reshape(self.dim_state, self.dim_state)
        R = np.asarray(R, dtype=float).reshape(self.dim_meas, self.dim_meas)

        H = self.measurement_matrix()
        z_pred = H @ x_col
        S = H @ P @ H.transpose() + R
        return z_pred[:, 0], S

    def project_extended(
        self,
        x: np.ndarray,
        P: np.ndarray,
        R: np.ndarray,
        sensor_model,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project state uncertainty for a nonlinear sensor model using its Jacobian."""

        x_col = _as_column(x, self.dim_state)
        P = np.asarray(P, dtype=float).reshape(self.dim_state, self.dim_state)
        R = np.asarray(R, dtype=float)
        dim_meas = R.shape[0]
        R = R.reshape(dim_meas, dim_meas)

        state = x_col[:, 0]
        H = np.asarray(sensor_model.calculate_jacobian(state), dtype=float).reshape(dim_meas, self.dim_state)
        z_pred = np.asarray(sensor_model.hx(state), dtype=float).reshape(dim_meas, 1)
        S = H @ P @ H.transpose() + R
        return z_pred[:, 0], S, H

    def update(
        self,
        x: np.ndarray,
        P: np.ndarray,
        z: np.ndarray,
        R: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Update state and covariance with a 3D position measurement."""

        x_col = _as_column(x, self.dim_state)
        z_col = _as_column(z, self.dim_meas)
        P = np.asarray(P, dtype=float).reshape(self.dim_state, self.dim_state)
        R = np.asarray(R, dtype=float).reshape(self.dim_meas, self.dim_meas)

        H = self.measurement_matrix()
        I = np.eye(self.dim_state, dtype=float)

        gamma = z_col - H @ x_col
        S = H @ P @ H.transpose() + R
        K = P @ H.transpose() @ np.linalg.inv(S)

        x_upd = x_col + K @ gamma
        P_upd = (I - K @ H) @ P
        return x_upd[:, 0], P_upd

    def update_extended(
        self,
        x: np.ndarray,
        P: np.ndarray,
        z: np.ndarray,
        R: np.ndarray,
        sensor_model,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extended Kalman filter update for a nonlinear sensor measurement."""

        x_col = _as_column(x, self.dim_state)
        z_col = np.asarray(z, dtype=float).reshape(-1, 1)
        P = np.asarray(P, dtype=float).reshape(self.dim_state, self.dim_state)
        I = np.eye(self.dim_state, dtype=float)

        z_pred, S, H = self.project_extended(x_col[:, 0], P, R, sensor_model)
        gamma = z_col - z_pred.reshape(-1, 1)
        K = P @ H.transpose() @ np.linalg.inv(S)

        x_upd = x_col + K @ gamma
        P_upd = (I - K @ H) @ P
        return x_upd[:, 0], P_upd
