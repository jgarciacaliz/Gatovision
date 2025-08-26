from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from typing import Optional
from .recognizer import CatRecognizer
from .memory import add_identity, rename_identity, list_identities

app = FastAPI(title="CatID", version="1.0.0")
recognizer = CatRecognizer()


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"ok": True}


@app.post("/infer")
async def infer(image: UploadFile = File(...)):
    """Infer the image and return detections."""
    data = await image.read()
    npimg = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    res = recognizer.process_image(img)
    return {"detections": res}


@app.get("/cats")
def cats():
    """Get the list of cats."""
    rows = list_identities()
    return {"cats": [{"id": r[0], "name": r[1]} for r in rows]}


@app.post("/cats/label")
def label_cat(temp_id: str = Form(...), name: str = Form(...)):
    """Label a cat with a real name."""
    rename_identity(temp_id, name)
    return {"ok": True, "id": temp_id, "name": name}


@app.post("/cats/add")
def add_cat(id: str = Form(...), name: str = Form(...)):
    """Add a new cat identity."""
    add_identity(id, name)
    return {"ok": True, "id": id, "name": name}
