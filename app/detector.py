from ultralytics import YOLO
import numpy as np
from typing import List, Tuple, Optional, Union
import torch
from .config import YOLO_MODEL_NAME, YOLO_CAT_CLASS_ID

class CatDetector:
    """Detector de gatos con YOLO, usa GPU + FP16 si hay y soporta batch."""
    def __init__(self, device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.device_index = 0 if device == "cuda" else None
        self.use_half = (device == "cuda")

        self.model = YOLO(YOLO_MODEL_NAME)
        try: self.model.to(self.device)
        except Exception: pass
        try: self.model.fuse()
        except Exception: pass

    def detect(
        self,
        img_bgr: np.ndarray,
        conf: float = 0.25,
        imgsz: int = 1080
    ) -> List[Tuple[int,int,int,int,float]]:
        """Detección para un solo frame."""
        res = self.model.predict(
            source=img_bgr,
            conf=conf,
            verbose=False,
            device=self.device_index if self.device == "cuda" else None,
            half=self.use_half,
            imgsz=imgsz
        )[0]
        out = []
        for b in res.boxes:
            if int(b.cls.item()) == YOLO_CAT_CLASS_ID:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                score = float(b.conf.item())
                out.append((x1, y1, x2, y2, score))
        return out

    def detect_batch(
        self,
        imgs_bgr: List[np.ndarray],
        conf: float = 0.25,
        imgsz: int = 1080
    ) -> List[List[Tuple[int,int,int,int,float]]]:
        """Detección para una lista de frames; devuelve lista por frame."""
        if not imgs_bgr:
            return []
        results = self.model.predict(
            source=imgs_bgr,
            conf=conf,
            verbose=False,
            device=self.device_index if self.device == "cuda" else None,
            half=self.use_half,
            imgsz=imgsz
        )
        batch_out: List[List[Tuple[int,int,int,int,float]]] = []
        for res in results:
            out = []
            for b in res.boxes:
                if int(b.cls.item()) == YOLO_CAT_CLASS_ID:
                    x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                    score = float(b.conf.item())
                    out.append((x1, y1, x2, y2, score))
            batch_out.append(out)
        return batch_out
