from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DetectionLimits:
    """Detection area limits used for BEV generation."""

    x: tuple[float, float] = (0.0, 50.0)
    y: tuple[float, float] = (-25.0, 25.0)
    z: tuple[float, float] = (-1.0, 3.0)
    intensity: tuple[float, float] = (0.0, 1.0)


@dataclass(frozen=True)
class BEVConfig:
    """Configuration for the LiDAR bird's-eye-view image."""

    bev_height: int = 608
    bev_width: int = 608
    limits: DetectionLimits = field(default_factory=DetectionLimits)


@dataclass(frozen=True)
class SFA3DConfig:
    """Minimal detector-facing configuration for project-side SFA3D integration."""

    bev: BEVConfig = field(default_factory=BEVConfig)
    external_repo_path: str | None = None
    checkpoint_path: str | None = None
    device: str = "cpu"
    score_threshold: float = 0.5
    arch: str = "fpn_resnet_18"
    imagenet_pretrained: bool = False
    head_conv: int = 64
    num_classes: int = 3
    num_center_offset: int = 2
    num_z: int = 1
    num_dim: int = 3
    num_direction: int = 2
    k: int = 50
    down_ratio: int = 4
    peak_thresh: float = 0.2
    class_labels: tuple[str, ...] = ("pedestrian", "car", "cyclist")

    @property
    def heads(self) -> dict[str, int]:
        """Prediction heads required by the external SFA3D model builder."""

        return {
            "hm_cen": self.num_classes,
            "cen_offset": self.num_center_offset,
            "direction": self.num_direction,
            "z_coor": self.num_z,
            "dim": self.num_dim,
        }


def default_bev_config() -> BEVConfig:
    """Return the default BEV configuration from the exercise notebook."""

    return BEVConfig()


def default_sfa3d_config() -> SFA3DConfig:
    """Return the default project-side SFA3D integration config."""

    return SFA3DConfig()
