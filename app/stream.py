import argparse
import time
from pathlib import Path
from typing import Union

import cv2
import numpy as np

# Importa tu reconocedor y utilidades
from .recognizer import CatRecognizer
from .utils import save_crop

# -------- Utilidades de dibujo --------

def draw_box_with_label(
    img: np.ndarray,
    bbox,
    label: str,
    color=(40, 180, 120),
    thickness: int = 2,
):
    """Dibuja una caja con etiqueta en la imagen."""
    x1, y1, x2, y2 = bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    # Fondo para etiqueta
    font = cv2.FONT_HERSHEY_SIMPLEX
    txt = label
    (tw, th), baseline = cv2.getTextSize(txt, font, 0.6, 2)
    th_box = th + baseline + 6
    cv2.rectangle(img, (x1, y1 - th_box), (x1 + tw + 8, y1), color, -1)
    cv2.putText(img, txt, (x1 + 4, y1 - 6), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

def put_hud(img: np.ndarray, fps: float, info: str = ""):
    """Coloca información en la esquina de la imagen (FPS, info adicional)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    txt = f"FPS: {fps:.1f}"
    cv2.putText(img, txt, (10, 24), font, 0.7, (30, 200, 255), 2, cv2.LINE_AA)
    if info:
        cv2.putText(img, info, (10, 50), font, 0.6, (220, 220, 220), 2, cv2.LINE_AA)

# -------- Captura de video --------

def open_capture(source: Union[int, str], width: int = 1280, height: int = 720) -> cv2.VideoCapture:
    """
    Abre webcam o stream.
    - En Windows, intenta CAP_DSHOW para reducir latencia en webcam.
    """
    if isinstance(source, int):
        # Webcam local
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(source)
        # Sugerimos resolución, puede que el driver no lo respete.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # suele mejorar FPS en Windows
    else:
        # URL (RTSP/HTTP/archivo)
        cap = cv2.VideoCapture(source)

    return cap

# -------- Main loop --------

def main():
    """Función principal para ejecutar el reconocimiento de gatos en tiempo real."""
    parser = argparse.ArgumentParser(description="CatID - detección y reconocimiento en tiempo real")
    parser.add_argument("--source", type=str, default="0", help="Índice de cámara (e.g., 0/1/2) o URL RTSP/HTTP/archivo")
    parser.add_argument("--conf", type=float, default=0.25, help="Confianza mínima de YOLO para detección de gatos")
    parser.add_argument("--view_size", type=str, default="", help='Ej: "1280x720" para redimensionar ventana de visualización')
    parser.add_argument("--save_dir", type=str, default="runs/crops", help="Carpeta para guardar fotogramas con tecla S")
    parser.add_argument("--show_boxes", action="store_true", help="Mostrar cajas (por defecto ON)", default=True)
    args = parser.parse_args()

    # Parseo de source
    try:
        src: Union[int, str] = int(args.source)
    except ValueError:
        src = args.source

    # Tamaño de visualización (opcional)
    view_w, view_h = None, None
    if "x" in args.view_size:
        try:
            w_str, h_str = args.view_size.lower().split("x")
            view_w, view_h = int(w_str), int(h_str)
        except Exception:
            print("[WARN] view_size inválido; ignoro argumento.")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Cargando modelos (YOLO + ResNet) ...")
    recognizer = CatRecognizer()  # carga una sola vez

    cap = open_capture(src)
    if not cap.isOpened():
        print(f"[ERROR] No pude abrir la fuente de video: {args.source}")
        return

    print("[INFO] Presiona Q para salir, S para guardar fotograma con detecciones.")
    prev_t = time.time()
    fps = 0.0
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Frame no disponible. Reintentando...")
            time.sleep(0.02)
            continue

        frame_idx += 1

        # Procesamiento
        results = recognizer.process_image(frame)

        # Dibujo
        draw_img = frame.copy()
        for r in results:
            x1, y1, x2, y2 = r["bbox"]
            if r["status"] == "known":
                label = f'{r["display_name"]}  (d={r["distance"]:.2f})'
                color = (40, 200, 100)
            else:
                label = f'{r["display_name"]} (NEW)'
                color = (80, 80, 230)
            draw_box_with_label(draw_img, (x1, y1, x2, y2), label, color=color)

        # FPS
        now = time.time()
        dt = now - prev_t
        prev_t = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)

        put_hud(draw_img, fps, info=f"Detections: {len(results)}")

        # Redimensionado para visualización si se pidió
        if view_w and view_h:
            draw_img = cv2.resize(draw_img, (view_w, view_h))

        cv2.imshow("CatID - Live", draw_img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            # Guardar fotograma y, opcionalmente, recortes
            out_path = save_dir / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(out_path), draw_img)
            print(f"[INFO] Guardado {out_path}")
            # Guarda también recortes por cada detección (útil para depurar memoria)
            for r in results:
                x1, y1, x2, y2 = r["bbox"]
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    save_crop(r["identity_id"], crop)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
