from dataclasses import dataclass


@dataclass(frozen=True)
class Detection3D:
    """Project-side 3D detection in vehicle/LiDAR-aligned coordinates."""

    x: float
    y: float
    z: float
    l: float
    w: float
    h: float
    yaw: float
    score: float
    label: str
