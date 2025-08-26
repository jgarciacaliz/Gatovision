# c:\Users\jgarc\OneDrive\Joe\Development\Repositories\Gatovision\main.py

import os
import re
import time
import argparse
import threading
import subprocess
from dataclasses import dataclass
from typing import Optional, Union, List, Tuple
from queue import Queue, Empty, Full

import numpy as np
import cv2
import torch
import av

from dotenv import load_dotenv

from app.recognizer import CatRecognizer

load_dotenv()

def getenv_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None: return default
    return v.strip().lower() in ("1","true","yes","on")

VIEW_W, VIEW_H = 1920, 1080
TARGET_FPS = 30
DETECT_INTERVAL_MS = 100
CONF_THRESH = 0.25
YOLO_IMGSZ = 640
FRAME_QUEUE_SIZE = 2
WATCHDOG_STALE_MS = 1500
USE_NVDEC = True

ENV_SOURCE = os.getenv("SOURCE", "").strip()

ENV_CAM_NAME = os.getenv("CAMERA_NAME").strip()
ENV_CAM_INDEX = os.getenv("CAMERA_INDEX", "").strip()

USE_NVDEC = getenv_bool("USE_NVDEC", True)

RTSP_OPTS = {
    "rtsp_transport": "tcp",
    "stimeout": "5000000",
    "max_delay": "300000",
    "fflags": "nobuffer",
    "flags": "low_delay",
    "probesize": "64K",
    "analyzeduration": "0",
}

DSHOW_OPTS = {
    "video_size": f"{VIEW_W}x{VIEW_H}",
    "framerate": str(TARGET_FPS),
    "rtbufsize": "256M",
    "fflags": "nobuffer",
}

def draw_box_with_label(img, bbox, label, color=(40, 180, 120), thickness=2):
    """Dibuja un cuadro delimitador con una etiqueta en la imagen."""
    x1, y1, x2, y2 = bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), bl = cv2.getTextSize(label, font, 0.6, 2)
    pad = 6
    y_top = max(0, y1 - th - bl - pad*2)
    cv2.rectangle(img, (x1, y_top), (x1 + tw + pad*2, y1), color, -1)
    cv2.putText(img, label, (x1 + pad, y1 - pad), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


def put_hud(img, fps, dets, stale_ms):
    """Coloca información en la esquina superior izquierda de la imagen."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, f"FPS: {fps:.1f}", (10, 24), font, 0.7, (30, 200, 255), 2, cv2.LINE_AA)
    cv2.putText(img, f"Detecciones: {dets}", (10, 50), font, 0.7, (220, 220, 220), 2, cv2.LINE_AA)
    if stale_ms is not None and stale_ms > 300:
        cv2.putText(img, f"STALENESS: {stale_ms} ms", (10, 76), font, 0.6, (60, 220, 60), 2, cv2.LINE_AA)


@dataclass
class SharedState:
    latest_frame: Optional[np.ndarray] = None
    latest_ts: float = 0.0
    latest_results: List[dict] = None
    det_busy: bool = False
    lock: threading.Lock = threading.Lock()


class BaseCapture:
    def start(self): ...
    def stop(self): ...
    def is_open(self) -> bool: ...
    def read(self) -> Optional[Tuple[np.ndarray, float]]: ...
    def reopen(self): ...


def _list_dshow_devices() -> List[str]:
    """
    Lista dispositivos de video con ffmpeg/dshow en Windows.
    """
    try:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", -1 * 0, "-f", "dshow", "-list_devices", "true", "-i", "dummy"],
            capture_output=True, text=True, check=False
        )
    except TypeError:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-f", "dshow", "-list_devices", "true", "-i", "dummy"],
            capture_output=True, text=True, check=False
        )
    out = proc.stderr
    devices = []
    in_video = False
    for line in out.splitlines():
        if "DirectShow video devices" in line:
            in_video = True
            continue
        if "DirectShow audio devices" in line:
            in_video = False
        if in_video:
            m = re.search(r'"([^"]+)"', line)
            if m:
                devices.append(m.group(1))
    return devices


def _pick_webcam_name() -> str:
    """
    Selección automática:
    1) Si CAMERA_NAME en .env => usarlo.
    2) Si CAMERA_INDEX en .env => mapear a nombre por listado dshow.
    3) Elegir el primer dispositivo de video no virtual (evita OBS Virtual Camera).
    """
    if ENV_CAM_NAME:
        return ENV_CAM_NAME

    names = _list_dshow_devices()
    if not names:
        raise RuntimeError("No pude listar dispositivos de video dshow. Asegúrate de tener FFmpeg en PATH.")

    # Por índice
    if ENV_CAM_INDEX != "":
        try:
            idx = int(ENV_CAM_INDEX)
            if 0 <= idx < len(names):
                return names[idx]
        except ValueError:
            pass

    # Elige el primero que no contenga "virtual" ni "OBS" (case-insensitive)
    for n in names:
        if re.search(r"virtual|obs", n, flags=re.I):
            continue
        return n

    # Si todos son virtuales, usa el primero igualmente
    return names[0]


class PyAVDShowCapture(BaseCapture):
    def __init__(self, device_name: str, width=VIEW_W, height=VIEW_H):
        self.device_name = device_name
        self.width = width
        self.height = height
        self.container: Optional[av.container.input.InputContainer] = None
        self.stream: Optional[av.video.stream.VideoStream] = None
        self._open()

    def _open(self):
        opts = dict(DSHOW_OPTS)
        opts["video_size"] = f"{self.width}x{self.height}"
        self.container = av.open(f"video={self.device_name}", format="dshow", options=opts)
        self.stream = self.container.streams.video[0]
        self.stream.thread_type = "AUTO"

    def is_open(self) -> bool:
        return self.container is not None

    def read(self) -> Optional[Tuple[np.ndarray, float]]:
        try:
            for packet in self.container.demux(self.stream):
                for frame in packet.decode():
                    img = frame.to_ndarray(format="bgr24")
                    if img.shape[1] != self.width or img.shape[0] != self.height:
                        img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
                    return img, time.time()
        except (StopIteration, av.AVError):
            return None
        except Exception:
            return None

    def reopen(self):
        try:
            if self.container:
                self.container.close()
        except Exception:
            pass
        self._open()

    def start(self): pass
    def stop(self):
        try:
            if self.container:
                self.container.close()
        except Exception:
            pass


class PyAVRtspCapture(BaseCapture):
    def __init__(self, url: str, width=VIEW_W, height=VIEW_H, use_nvdec=True):
        self.url = url
        self.width = width
        self.height = height
        self.use_nvdec = use_nvdec
        self.container = None
        self.stream = None
        self._open()

    def _open(self):
        opts = dict(RTSP_OPTS)
        if self.use_nvdec:
            opts["hwaccel"] = "cuda"
        self.container = av.open(self.url, options=opts)
        self.stream = self.container.streams.video[0]
        self.stream.thread_type = "AUTO"

    def is_open(self) -> bool:
        return self.container is not None

    def read(self) -> Optional[Tuple[np.ndarray, float]]:
        try:
            for packet in self.container.demux(self.stream):
                for frame in packet.decode():
                    img = frame.to_ndarray(format="bgr24")
                    if img.shape[1] != self.width or img.shape[0] != self.height:
                        img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
                    return img, time.time()
        except (StopIteration, av.AVError):
            return None
        except Exception:
            return None

    def reopen(self):
        try:
            if self.container:
                self.container.close()
        except Exception:
            pass
        self._open()

    def start(self): pass
    def stop(self):
        try:
            if self.container:
                self.container.close()
        except Exception:
            pass


@dataclass
class SharedState:
    latest_frame: Optional[np.ndarray] = None
    latest_ts: float = 0.0
    latest_results: List[dict] = None
    det_busy: bool = False
    lock: threading.Lock = threading.Lock()


class CaptureWorker(threading.Thread):
    def __init__(self, cap: BaseCapture, q: Queue, state: SharedState):
        super().__init__(daemon=True)
        self.q = q
        self.state = state
        self.cap = cap
        self.stop_flag = False

    def reopen(self):
        try:
            self.cap.reopen()
        except Exception as e:
            print(f"[WARN] Reopen fallo: {e}")

    def run(self):
        while not self.stop_flag:
            out = self.cap.read()
            if out is None:
                time.sleep(0.002)
                continue
            frame, ts = out
            try:
                while True:
                    self.q.put_nowait((frame, ts))
                    break
            except Full:
                try:
                    _ = self.q.get_nowait()
                except Empty:
                    pass
                try:
                    self.q.put_nowait((frame, ts))
                except Full:
                    pass

            with self.state.lock:
                self.state.latest_frame = frame
                self.state.latest_ts = ts

        self.cap.stop()

    def stop(self):
        self.stop_flag = True


class SchedulerWorker(threading.Thread):
    def __init__(self, interval_ms: int, event: threading.Event, state: SharedState):
        super().__init__(daemon=True)
        self.interval = interval_ms / 1000.0
        self.event = event
        self.state = state
        self.stop_flag = False

    def run(self):
        next_t = time.time() + self.interval
        while not self.stop_flag:
            now = time.time()
            if now >= next_t:
                with self.state.lock:
                    busy = self.state.det_busy
                if not busy:
                    self.event.set()
                next_t += self.interval
            else:
                time.sleep(min(0.001, next_t - now))

    def stop(self):
        self.stop_flag = True


class InferWorker(threading.Thread):
    def __init__(self, q_frames: Queue, state: SharedState):
        super().__init__(daemon=True)
        self.q_frames = q_frames
        self.state = state
        self.stop_flag = False
        self.tick = threading.Event()
        self.rec: Optional[CatRecognizer] = None

    def start(self):
        torch.set_num_threads(1)
        print("[INFO] Cargando modelos (GPU FP16)…")
        self.rec = CatRecognizer()
        _ = self.rec.detector.detect(np.zeros((VIEW_H, VIEW_W, 3), dtype=np.uint8), conf=0.10, imgsz=YOLO_IMGSZ)
        self.scheduler = SchedulerWorker(DETECT_INTERVAL_MS, self.tick, self.state)
        self.scheduler.start()
        super().start()

    def run(self):
        while not self.stop_flag:
            fired = self.tick.wait(timeout=0.2)
            if not fired:
                continue
            self.tick.clear()

            with self.state.lock:
                self.state.det_busy = True

            frame = None
            ts = None
            try:
                while True:
                    frame, ts = self.q_frames.get_nowait()
            except Empty:
                with self.state.lock:
                    frame = self.state.latest_frame
                    ts = self.state.latest_ts

            if frame is None:
                with self.state.lock:
                    self.state.det_busy = False
                continue

            try:
                outs_batch = self.rec.process_batch(
                    [frame],
                    realtime=True,
                    update_memory=False,
                    save_crops=False,
                    conf=CONF_THRESH,
                    imgsz=YOLO_IMGSZ
                )
                outs = outs_batch[0] if outs_batch else []
            except Exception as e:
                print(f"[WARN] Infer error: {e}")
                outs = []

            with self.state.lock:
                self.state.latest_results = outs
                self.state.det_busy = False

    def stop(self):
        self.stop_flag = True
        if hasattr(self, "scheduler"):
            self.scheduler.stop()


def run():
    cv2.setUseOptimized(True)

    cap: BaseCapture
    if ENV_SOURCE and (ENV_SOURCE.startswith("rtsp://") or ENV_SOURCE.startswith("http")):
        print("[INFO] Fuente: RTSP/HTTP vía PyAV")
        cap = PyAVRtspCapture(ENV_SOURCE, use_nvdec=USE_NVDEC)
    else:
        cam_name = _pick_webcam_name()
        print(f"[INFO] Fuente: Webcam vía PyAV (dshow) → \"{cam_name}\"")
        cap = PyAVDShowCapture(cam_name)

    state = SharedState(latest_results=[])
    q_frames: Queue = Queue(maxsize=FRAME_QUEUE_SIZE)

    capw = CaptureWorker(cap, q_frames, state)
    capw.start()

    infw = InferWorker(q_frames, state)
    infw.start()

    print(f"[INFO] Ventana lista.  Q=salir, S=snapshot. Objetivo: {TARGET_FPS} FPS")
    fps = 0.0
    prev_t = time.time()
    snaps = 0

    while True:
        frame = None
        ts = None
        try:
            while True:
                frame, ts = q_frames.get_nowait()
        except Empty:
            with state.lock:
                frame = state.latest_frame
                ts = state.latest_ts

        if frame is None:
            frame = np.zeros((VIEW_H, VIEW_W, 3), dtype=np.uint8)

        with state.lock:
            results = state.latest_results[:] if state.latest_results is not None else []

        draw = frame.copy()
        for r in results:
            x1, y1, x2, y2 = r["bbox"]
            if r["status"] == "known":
                label = f'{r["display_name"]} (d={r["distance"]:.2f})'
                color = (40, 200, 100)
            else:
                label = f'{r["display_name"]} (NEW)'
                color = (80, 80, 230)
            draw_box_with_label(draw, (x1, y1, x2, y2), label, color=color)

        stale_ms = None
        if ts:
            stale_ms = int((time.time() - ts) * 1000)
            if stale_ms > WATCHDOG_STALE_MS:
                print(f"[WARN] Fuente estancada ({stale_ms} ms). Reabriendo…")
                capw.reopen()

        now = time.time()
        dt = now - prev_t
        prev_t = now
        inst = 1.0 / dt if dt > 0 else TARGET_FPS
        fps = 0.9 * fps + 0.1 * inst if fps > 0 else inst
        put_hud(draw, fps, len(results), stale_ms)

        cv2.imshow("CatID — 1080p Realtime (PyAV webcam/RTSP)", draw)
        delay = max(1.0 / TARGET_FPS - (time.time() - now), 0)
        key = cv2.waitKey(int(delay * 1000)) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            os.makedirs("runs/crops", exist_ok=True)
            snaps += 1
            out = os.path.join("runs/crops", f"snapshot_{snaps:03d}.jpg")
            cv2.imwrite(out, draw)
            print(f"[INFO] Guardado {out}")

    capw.stop()
    infw.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CatID — 1080p@30 con PyAV (webcam dshow y RTSP) + .env")
    _ = parser.parse_args()
    run()
