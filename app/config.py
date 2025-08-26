from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MEMORY_DB = DATA_DIR / "cats.db"
MEMORY_IMAGES = DATA_DIR / "mem_images"  # donde guardamos recortes por identidad

YOLO_MODEL_NAME = "yolov8n.pt"
YOLO_CAT_CLASS_ID = 15  # en COCO, 'cat' = 15

EMBED_BACKBONE = "resnet50"  # fácil de usar sin pesos extra
EMBED_SIZE = 2048

MATCH_THRESHOLD = 0.25

MIN_SAMPLES_PER_ID = 3

RANDOM_SEED = 42
