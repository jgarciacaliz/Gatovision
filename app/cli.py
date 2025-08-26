import argparse
from pathlib import Path
import cv2
import glob
from .recognizer import CatRecognizer
from .memory import add_identity, rename_identity

def run_on_folder(folder: Path):
    """Procesa todas las imágenes en la carpeta dada."""
    rec = CatRecognizer()
    images = sorted(glob.glob(str(folder / "*.*")))
    for p in images:
        img = cv2.imread(p)
        if img is None: 
            print(f"[WARN] No pude leer {p}")
            continue
        res = rec.process_image(img)
        print(f"\nImagen: {Path(p).name}")
        for r in res:
            print(r)

def enroll_initial(name: str, folder: Path):
    """Enrola fotos iniciales de un gato."""
    rec = CatRecognizer()
    add_identity(name, name)
    images = sorted(glob.glob(str(folder / "*.*")))
    for p in images:
        img = cv2.imread(p)
        if img is None: 
            continue
        results = rec.process_image(img)
        for r in results:
            if r["status"] == "new":
                rename_identity(r["identity_id"], name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, help="Procesa todas las imágenes del folder")
    parser.add_argument("--enroll", nargs=2, metavar=("NAME","FOLDER"),
                        help="Enrola fotos iniciales de un gato: --enroll Mishi ./data/mishi_fotos")
    args = parser.parse_args()

    if args.enroll:
        name, folder = args.enroll
        enroll_initial(name, Path(folder))
    elif args.folder:
        run_on_folder(Path(args.folder))
    else:
        print("Usos:")
        print("  python -m app.cli --enroll Mishi ./data/mishi_fotos")
        print("  python -m app.cli --folder ./data/test")
