import numpy as np
import cv2
import uuid
from typing import Dict, Any, List
from .detector import CatDetector
from .embeddings import Embedder
from .memory import (
    init_db, add_identity, add_sample, get_all_identities_with_embs, get_prototype
)
from .config import MATCH_THRESHOLD
from .utils import crop_from_bbox, to_rgb, normalize_imagenet, save_crop


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - np.dot(a, b))


class CatRecognizer:
    def __init__(self):
        init_db()
        self.detector = CatDetector()
        self.embedder = Embedder()

    def _best_match(self, emb: np.ndarray) -> Dict[str, Any]:
        catalog = get_all_identities_with_embs()
        best_id, best_name, best_d = None, None, 1e9
        for ident_id, name, embs in catalog:
            proto = get_prototype(embs)
            if proto is None:
                continue
            d = cosine_distance(emb, proto)
            if d < best_d:
                best_id, best_name, best_d = ident_id, name, d
        return {"identity_id": best_id, "display_name": best_name, "distance": best_d}

    def process_image(
        self,
        img_bgr,
        realtime: bool = False,
        update_memory: bool = True,
        save_crops: bool = True,
        conf: float = 0.25,
        imgsz: int = 1080
    ) -> List[Dict[str, Any]]:
        dets = self.detector.detect(img_bgr, conf=conf, imgsz=imgsz)
        return self._postprocess(img_bgr, dets, realtime, update_memory, save_crops)

    def process_batch(
        self,
        imgs_bgr: List[np.ndarray],
        realtime: bool = False,
        update_memory: bool = True,
        save_crops: bool = True,
        conf: float = 0.25,
        imgsz: int = 1080
    ) -> List[List[Dict[str, Any]]]:
        batch_dets = self.detector.detect_batch(imgs_bgr, conf=conf, imgsz=imgsz)
        outs: List[List[Dict[str, Any]]] = []
        for img, dets in zip(imgs_bgr, batch_dets):
            outs.append(self._postprocess(img, dets, realtime, update_memory, save_crops))
        return outs

    def _postprocess(
        self,
        img_bgr,
        dets: List[tuple],
        realtime: bool,
        update_memory: bool,
        save_crops: bool
    ) -> List[Dict[str, Any]]:
        results = []
        crops_norm = []
        crops_meta = []
        for (x1, y1, x2, y2, score) in dets:
            crop = crop_from_bbox(img_bgr, (x1, y1, x2, y2))
            if crop.size == 0:
                continue
            rgb = to_rgb(crop)
            norm = normalize_imagenet(rgb, 224)
            crops_norm.append(norm)
            crops_meta.append(((x1, y1, x2, y2), crop))

        if not crops_norm:
            return results

        embs = self.embedder.encode_batch(np.stack(crops_norm, axis=0))
        for emb, (bbox, crop) in zip(embs, crops_meta):
            match = self._best_match(emb)
            is_new = (match["identity_id"] is None) or (match["distance"] > MATCH_THRESHOLD)
            if realtime:
                results.append({
                    "bbox": list(bbox),
                    "identity_id": match["identity_id"] if not is_new else "unknown",
                    "display_name": match["display_name"] if not is_new else "unknown",
                    "distance": None if is_new else float(match["distance"]),
                    "status": "new" if is_new else "known"
                })
            else:
                if is_new:
                    new_id = f"unknown-{uuid.uuid4().hex[:8]}"
                    add_identity(new_id, new_id)
                    add_sample(new_id, emb, meta={"source": "auto", "note": "auto-enroll"})
                    if save_crops:
                        save_crop(new_id, crop)
                    results.append({
                        "bbox": list(bbox),
                        "identity_id": new_id,
                        "display_name": new_id,
                        "distance": None,
                        "status": "new"
                    })
                else:
                    if update_memory:
                        add_sample(match["identity_id"], emb, meta={"source": "trace"})
                        if save_crops:
                            save_crop(match["identity_id"], crop)
                    results.append({
                        "bbox": list(bbox),
                        "identity_id": match["identity_id"],
                        "display_name": match["display_name"],
                        "distance": float(match["distance"]),
                        "status": "known"
                    })
        return results
