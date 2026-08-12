import numpy as np
from typing import Any, Dict

from . import UltralyticsYOLOModel
from app.core.registry import register_model
from app.schemas.shape import Shape


@register_model("yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x")
class YOLO26nDetection(UltralyticsYOLOModel):
    """YOLO26 object detection model."""

    def load(self):
        """Load YOLO26 model."""
        from ultralytics import YOLO

        model_path = self.params.get("model_path", "yolo26n.pt")
        device = self.params.get("device", "cpu")

        self.model = YOLO(model_path)
        self.model.to(device)

        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model(dummy_img, verbose=False)

    def predict(
        self, image: np.ndarray, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute object detection."""
        conf_threshold = params.get(
            "conf_threshold", self.params.get("conf_threshold", 0.25)
        )
        iou_threshold = params.get(
            "iou_threshold", self.params.get("iou_threshold", 0.45)
        )

        results = self.model(
            image,
            conf=conf_threshold,
            iou=iou_threshold,
            classes=self.get_filter_class_ids(params),
            verbose=False,
        )

        shapes = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = result.names[cls]

                    shape = Shape(
                        label=label,
                        shape_type="rectangle",
                        points=[
                            [float(xyxy[0]), float(xyxy[1])],
                            [float(xyxy[2]), float(xyxy[1])],
                            [float(xyxy[2]), float(xyxy[3])],
                            [float(xyxy[0]), float(xyxy[3])],
                        ],
                        score=conf,
                    )
                    shapes.append(shape)

        return {"shapes": shapes, "description": ""}

    def unload(self):
        """Release model resources."""
        if hasattr(self, "model"):
            del self.model