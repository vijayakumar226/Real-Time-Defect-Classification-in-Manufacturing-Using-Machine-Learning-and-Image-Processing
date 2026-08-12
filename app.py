import os
import uuid
import shutil
import cv2

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO

app = FastAPI(title="Defect Classification")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# Serve CSS + output images
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

templates = Jinja2Templates(directory=TEMPLATE_DIR)

# Load model once
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")
model = YOLO(MODEL_PATH)

def run_detection_and_save(input_image_path: str, prefix: str):
    results = model(input_image_path)
    result = results[0]

    annotated = result.plot()
    out_name = f"{prefix}{uuid.uuid4().hex}.jpg"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    cv2.imwrite(out_path, annotated)

    detections = []
    for box in result.boxes:
        cls_id = int(box.cls[0].item())
        label = model.names[cls_id]
        conf = float(box.conf[0].item())
        detections.append({"label": label, "confidence": round(conf, 3)})

    return detections, f"/outputs/{out_name}"

# -------------------------
# Pages
# -------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/webcam", response_class=HTMLResponse)
def webcam_page(request: Request):
    return templates.TemplateResponse("webcam.html", {"request": request})

@app.get("/upload", response_class=HTMLResponse)
def upload_page(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

# -------------------------
# APIs used by pages
# -------------------------
@app.post("/detect_frame")
async def detect_frame(file: UploadFile = File(...)):
    # Accept webcam frame image
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        return JSONResponse({"error": "Frame must be JPG/PNG image."}, status_code=400)

    ext = os.path.splitext(file.filename)[1].lower()
    frame_name = f"frame_{uuid.uuid4().hex}{ext}"
    frame_path = os.path.join(UPLOAD_DIR, frame_name)

    with open(frame_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    detections, output_url = run_detection_and_save(frame_path, prefix="cam_")
    return {"detections": detections, "output_url": output_url}

@app.post("/detect_upload")
async def detect_upload(file: UploadFile = File(...)):
    # Accept upload image
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        return JSONResponse({"error": "Please upload JPG/PNG image."}, status_code=400)

    ext = os.path.splitext(file.filename)[1].lower()
    up_name = f"upload_{uuid.uuid4().hex}{ext}"
    up_path = os.path.join(UPLOAD_DIR, up_name)

    with open(up_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    detections, output_url = run_detection_and_save(up_path, prefix="upl_")
    return {"detections": detections, "output_url": output_url}
