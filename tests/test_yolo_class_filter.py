import unittest
from pathlib import Path
from typing import Any, Dict, List, Tuple, Type

import numpy as np
import yaml

from app.models import UltralyticsYOLOModel
from app.models.yolo11 import YOLO11Detection
from app.models.yolo11_obb import YOLO11OBB
from app.models.yolo11_pose import YOLO11Pose
from app.models.yolo11_seg import YOLO11Segmentation
from app.models.yolo11_track import YOLO11DetectionTrack
from app.models.yolo26n import YOLO26nDetection


class FakeYOLO:
    """Capture Ultralytics inference arguments for class-filter tests."""

    def __init__(self) -> None:
        self.names = {0: "person", 1: "car", 2: "dog"}
        self.kwargs: Dict[str, Any] = {}

    def __call__(self, *args: Any, **kwargs: Any) -> List[Any]:
        """Capture regular inference arguments.

        Args:
            *args: Positional inference arguments.
            **kwargs: Keyword inference arguments.

        Returns:
            Empty prediction results.
        """
        self.kwargs = kwargs
        return []

    def track(self, *args: Any, **kwargs: Any) -> List[Any]:
        """Capture tracking inference arguments.

        Args:
            *args: Positional inference arguments.
            **kwargs: Keyword inference arguments.

        Returns:
            Empty tracking results.
        """
        self.kwargs = kwargs
        return []


class TestYOLOClassFilter(unittest.TestCase):
    def _create_model(
        self, model_class: Type[UltralyticsYOLOModel]
    ) -> Tuple[UltralyticsYOLOModel, FakeYOLO]:
        """Create an unloaded model with a fake Ultralytics backend.

        Args:
            model_class: YOLO model implementation class.

        Returns:
            Model instance and its fake inference backend.
        """
        model = model_class(
            {
                "model_id": "test_yolo",
                "display_name": "Test YOLO",
                "params": {"filter_classes": ["person", "dog"]},
            }
        )
        backend = FakeYOLO()
        model.model = backend
        model.tracker_type = "bytetrack"
        return model, backend

    def test_metadata_exposes_classes_and_default_filter(self) -> None:
        model, _ = self._create_model(YOLO11Detection)

        metadata = model.get_metadata()

        self.assertEqual(metadata["classes"], ["person", "car", "dog"])
        self.assertEqual(metadata["filter_classes"], ["person", "dog"])
        self.assertIsNone(model.get_filter_class_ids({"filter_classes": []}))

    def test_all_yolo_tasks_forward_selected_class_ids(self) -> None:
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        model_classes = (
            YOLO11Detection,
            YOLO11Segmentation,
            YOLO11Pose,
            YOLO11OBB,
            YOLO11DetectionTrack,
            YOLO26nDetection,
        )

        for model_class in model_classes:
            with self.subTest(model_class=model_class.__name__):
                model, backend = self._create_model(model_class)
                model.predict(image, {"filter_classes": ["car"]})

                self.assertEqual(backend.kwargs["classes"], [1])

    def test_yolo_configs_advertise_class_filter(self) -> None:
        config_dir = (
            Path(__file__).resolve().parents[1] / "configs" / "auto_labeling"
        )

        for config_path in config_dir.glob("yolo11*.yaml"):
            with self.subTest(config_path=config_path.name):
                with open(config_path, "r") as config_file:
                    config = yaml.safe_load(config_file)
                widget_names = {
                    widget["name"] for widget in config.get("widgets", [])
                }

                self.assertIn("button_classes_filter", widget_names)


if __name__ == "__main__":
    unittest.main()
