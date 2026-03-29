from __future__ import annotations

from dataclasses import dataclass, field
import importlib
from pathlib import Path
from types import SimpleNamespace
import sys
from typing import Any

import numpy as np

from detection.types import Detection3D

from .bev import pcl_to_bev
from .config import SFA3DConfig, default_sfa3d_config


@dataclass
class SFA3DAdapter:
    """Thin project-side boundary around BEV generation and SFA3D model loading."""

    config: SFA3DConfig = field(default_factory=default_sfa3d_config)
    model: Any = field(init=False, default=None, repr=False)
    device: Any = field(init=False, default=None, repr=False)
    raw_outputs: dict[str, Any] | None = field(init=False, default=None, repr=False)
    decoded_outputs: list[dict[int, np.ndarray]] | None = field(init=False, default=None, repr=False)
    _create_model: Any = field(init=False, default=None, repr=False)
    _sigmoid: Any = field(init=False, default=None, repr=False)
    _decode: Any = field(init=False, default=None, repr=False)
    _post_processing: Any = field(init=False, default=None, repr=False)
    _external_sfa_path: Path | None = field(init=False, default=None, repr=False)
    model_loaded: bool = field(init=False, default=False)

    def _validate_paths(self) -> tuple[Path, Path]:
        """Validate the external SFA3D repository root and checkpoint path."""

        if not self.config.external_repo_path:
            raise ValueError("SFA3D config is missing 'external_repo_path'.")
        if not self.config.checkpoint_path:
            raise ValueError("SFA3D config is missing 'checkpoint_path'.")

        repo_root = Path(self.config.external_repo_path).expanduser().resolve()
        if not repo_root.is_dir():
            raise FileNotFoundError(f"SFA3D repository path does not exist: {repo_root}")

        sfa_path = repo_root / "sfa"
        if not sfa_path.is_dir():
            raise FileNotFoundError(
                f"Expected an 'sfa' directory inside the SFA3D repository root: {repo_root}"
            )

        model_utils_path = sfa_path / "models" / "model_utils.py"
        if not model_utils_path.is_file():
            raise FileNotFoundError(
                f"Could not find SFA3D model builder at: {model_utils_path}"
            )

        checkpoint_path = Path(self.config.checkpoint_path).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"SFA3D checkpoint file does not exist: {checkpoint_path}")

        return sfa_path, checkpoint_path

    def _import_external_modules(self) -> None:
        """Import the minimum external SFA3D modules needed for model construction and decoding."""

        if self._external_sfa_path is None:
            raise RuntimeError("SFA3D external path is not set. Call _validate_paths first.")

        sfa_path_str = str(self._external_sfa_path)
        if sfa_path_str not in sys.path:
            sys.path.append(sfa_path_str)

        try:
            model_utils = importlib.import_module("models.model_utils")
        except ModuleNotFoundError as exc:
            if exc.name == "torch":
                raise ImportError(
                    "PyTorch is not installed in the active Python environment. "
                    "SFA3D model loading requires torch before external modules can be imported."
                ) from exc
            raise ImportError(
                f"Missing Python dependency while importing SFA3D model modules from: {self._external_sfa_path}"
            ) from exc
        except Exception as exc:
            raise ImportError(
                f"Failed to import SFA3D model utilities from: {self._external_sfa_path}"
            ) from exc

        try:
            torch_utils = importlib.import_module("utils.torch_utils")
            evaluation_utils = importlib.import_module("utils.evaluation_utils")
        except ModuleNotFoundError as exc:
            if exc.name == "torch":
                raise ImportError(
                    "PyTorch is not installed in the active Python environment. "
                    "SFA3D decode utilities require torch."
                ) from exc
            raise ImportError(
                f"Missing Python dependency while importing SFA3D decode utilities from: {self._external_sfa_path}"
            ) from exc
        except Exception as exc:
            raise ImportError(
                f"Failed to import SFA3D decode utilities from: {self._external_sfa_path}"
            ) from exc

        self._create_model = model_utils.create_model
        self._sigmoid = torch_utils._sigmoid
        self._decode = evaluation_utils.decode
        self._post_processing = evaluation_utils.post_processing

    def _build_model(self) -> Any:
        """Build the external SFA3D model using the minimal config it expects."""

        if self._create_model is None:
            raise RuntimeError("SFA3D model builder is not imported. Call _import_external_modules first.")

        external_config = SimpleNamespace(
            arch=self.config.arch,
            heads=self.config.heads,
            head_conv=self.config.head_conv,
            imagenet_pretrained=self.config.imagenet_pretrained,
        )
        return self._create_model(external_config)

    def _resolve_device(self) -> Any:
        """Resolve the configured torch device with a clear CUDA error message."""

        try:
            import torch
        except ModuleNotFoundError as exc:
            raise ImportError(
                "PyTorch is not installed in the active Python environment. "
                "SFA3D model loading requires torch."
            ) from exc

        if self.config.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device '{self.config.device}' was requested, but CUDA is not available."
            )
        return torch.device(self.config.device)

    def _ensure_model_loaded(self) -> None:
        """Raise a clear error if the caller tries to run the model before loading it."""

        if not self.model_loaded or self.model is None or self.device is None:
            raise RuntimeError("SFA3D model is not loaded. Call load_model() before running a forward pass.")

    def _ensure_decode_utils_loaded(self) -> None:
        """Raise a clear error if decode utilities are not available."""

        if self._sigmoid is None or self._decode is None or self._post_processing is None:
            raise RuntimeError("SFA3D decode utilities are not loaded. Call load_model() first.")

    def load_model(self) -> None:
        """Validate paths, import SFA3D, build the model, load weights, and set eval mode."""

        sfa_path, checkpoint_path = self._validate_paths()
        self._external_sfa_path = sfa_path
        self._import_external_modules()

        model = self._build_model()

        try:
            import torch
        except ModuleNotFoundError as exc:
            raise ImportError(
                "PyTorch is not installed in the active Python environment. "
                "SFA3D model loading requires torch."
            ) from exc

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)

        device = self._resolve_device()
        model = model.to(device=device)
        model.eval()

        self.model = model
        self.device = device
        self.model_loaded = True
        self.raw_outputs = None
        self.decoded_outputs = None

    def prepare_bev(self, pcl: np.ndarray) -> np.ndarray:
        """Convert a decoded LiDAR point cloud into the BEV input used by SFA3D."""

        return pcl_to_bev(pcl, self.config.bev)

    def prepare_input_tensor(self, bev_map: np.ndarray) -> Any:
        """Convert BEV numpy data from (3, H, W) to a float32 torch tensor (1, 3, H, W)."""

        self._ensure_model_loaded()

        if bev_map.ndim != 3 or bev_map.shape[0] != 3:
            raise ValueError("BEV map must have shape (3, H, W).")

        try:
            import torch
        except ModuleNotFoundError as exc:
            raise ImportError(
                "PyTorch is not installed in the active Python environment. "
                "SFA3D forward pass requires torch."
            ) from exc

        bev_map = np.ascontiguousarray(bev_map, dtype=np.float32)
        input_tensor = torch.from_numpy(bev_map).unsqueeze(0).float().to(self.device)
        return input_tensor

    def forward_from_bev(self, bev_map: np.ndarray) -> dict[str, Any]:
        """Run one raw forward pass on a prepared BEV map and return the output dictionary."""

        self._ensure_model_loaded()
        input_tensor = self.prepare_input_tensor(bev_map)

        try:
            import torch
        except ModuleNotFoundError as exc:
            raise ImportError(
                "PyTorch is not installed in the active Python environment. "
                "SFA3D forward pass requires torch."
            ) from exc

        with torch.no_grad():
            outputs = self.model(input_tensor)

        if not isinstance(outputs, dict):
            raise TypeError(f"Expected SFA3D model to return a dict, got: {type(outputs)}")

        self.raw_outputs = outputs
        return outputs

    def forward_from_pcl(self, pcl: np.ndarray) -> dict[str, Any]:
        """Prepare BEV from a decoded LiDAR point cloud and run one raw forward pass."""

        bev_map = self.prepare_bev(pcl)
        return self.forward_from_bev(bev_map)

    def describe_output_shapes(self, outputs: dict[str, Any] | None = None) -> dict[str, tuple[int, ...]]:
        """Return a simple mapping of raw output keys to tensor shapes."""

        if outputs is None:
            outputs = self.raw_outputs
        if outputs is None:
            raise RuntimeError("No raw SFA3D outputs are available yet. Run forward_from_bev() first.")

        return {key: tuple(value.shape) for key, value in outputs.items()}

    def decode_raw_outputs(self, outputs: dict[str, Any] | None = None) -> list[dict[int, np.ndarray]]:
        """Apply sigmoid, decode, and post-processing to the current raw SFA3D outputs."""

        self._ensure_model_loaded()
        self._ensure_decode_utils_loaded()

        if outputs is None:
            outputs = self.raw_outputs
        if outputs is None:
            raise RuntimeError("No raw SFA3D outputs are available yet. Run forward_from_bev() first.")

        required_keys = ("hm_cen", "cen_offset", "direction", "z_coor", "dim")
        missing = [key for key in required_keys if key not in outputs]
        if missing:
            raise KeyError(f"Missing required SFA3D output heads for decode: {missing}")

        hm_cen = self._sigmoid(outputs["hm_cen"].clone())
        cen_offset = self._sigmoid(outputs["cen_offset"].clone())
        direction = outputs["direction"].clone()
        z_coor = outputs["z_coor"].clone()
        dim = outputs["dim"].clone()

        detections = self._decode(
            hm_cen,
            cen_offset,
            direction,
            z_coor,
            dim,
            K=self.config.k,
        )
        detections = detections.cpu().numpy().astype(np.float32)
        decoded = self._post_processing(
            detections,
            self.config.num_classes,
            self.config.down_ratio,
            self.config.peak_thresh,
        )

        self.decoded_outputs = decoded
        return decoded

    def decode_from_bev(self, bev_map: np.ndarray) -> list[dict[int, np.ndarray]]:
        """Run a forward pass on a BEV map and return the decoded/postprocessed structure."""

        outputs = self.forward_from_bev(bev_map)
        return self.decode_raw_outputs(outputs)

    def decode_from_pcl(self, pcl: np.ndarray) -> list[dict[int, np.ndarray]]:
        """Run BEV generation, forward pass, and decode for one decoded LiDAR point cloud."""

        outputs = self.forward_from_pcl(pcl)
        return self.decode_raw_outputs(outputs)

    def _decode_row_to_detection(self, cls_id: int, row: np.ndarray) -> Detection3D:
        """Convert one decoded SFA3D row into a metric Detection3D.

        External SFA3D post_processing returns rows ordered as:
        [score, x_bev_px, y_bev_px, z_rel, h_m, w_bev_px, l_bev_px, yaw_bev]

        The x/y center and w/l values are still in BEV pixel space. This method mirrors
        the external repo's convert_det_to_real_values(...) logic, but uses this project's
        BEV limits so the returned detection is expressed in project metric coordinates.
        """

        if cls_id < 0 or cls_id >= len(self.config.class_labels):
            raise ValueError(f"Unknown SFA3D class id: {cls_id}")

        score, x_bev_px, y_bev_px, z_rel, h_m, w_bev_px, l_bev_px, yaw_bev = [float(value) for value in row]

        limits = self.config.bev.limits
        bound_size_x = limits.x[1] - limits.x[0]
        bound_size_y = limits.y[1] - limits.y[0]

        x_m = y_bev_px / self.config.bev.bev_height * bound_size_x + limits.x[0]
        y_m = x_bev_px / self.config.bev.bev_width * bound_size_y + limits.y[0]
        z_m = z_rel + limits.z[0]
        w_m = w_bev_px / self.config.bev.bev_width * bound_size_y
        l_m = l_bev_px / self.config.bev.bev_height * bound_size_x
        yaw = -yaw_bev

        return Detection3D(
            x=x_m,
            y=y_m,
            z=z_m,
            l=l_m,
            w=w_m,
            h=h_m,
            yaw=yaw,
            score=score,
            label=self.config.class_labels[cls_id],
        )

    def convert_decoded_outputs(
        self,
        decoded: list[dict[int, np.ndarray]] | None = None,
    ) -> list[Detection3D]:
        """Flatten one-frame decoded SFA3D output into project-side Detection3D objects.

        Assumes the caller is working with a single frame. If more than one frame is present,
        the first batch element is converted.
        """

        if decoded is None:
            decoded = self.decoded_outputs
        if decoded is None:
            raise RuntimeError("No decoded SFA3D outputs are available yet. Run decode_raw_outputs() first.")
        if len(decoded) == 0:
            return []

        frame_decoded = decoded[0]
        detections: list[Detection3D] = []

        for cls_id, rows in frame_decoded.items():
            if rows is None or len(rows) == 0:
                continue
            for row in rows:
                detection = self._decode_row_to_detection(int(cls_id), row)
                if detection.score >= self.config.score_threshold:
                    detections.append(detection)

        return detections

    def infer_from_bev(self, bev_map: np.ndarray) -> list[Detection3D]:
        """Run decode/post-processing and convert the first-frame result to Detection3D objects."""

        decoded = self.decode_from_bev(bev_map)
        return self.convert_decoded_outputs(decoded)

    def detect(self, pcl: np.ndarray) -> list[Detection3D]:
        """End-to-end detector entry point for one decoded LiDAR point cloud."""

        decoded = self.decode_from_pcl(pcl)
        return self.convert_decoded_outputs(decoded)


def build_sfa3d_adapter(config: SFA3DConfig | None = None) -> SFA3DAdapter:
    """Create the default thin adapter used for future detector integration."""

    if config is None:
        config = default_sfa3d_config()
    return SFA3DAdapter(config=config)
