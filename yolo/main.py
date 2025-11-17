from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image, ImageDraw
import io
import os
from datetime import datetime

app = FastAPI()

# CORS ayarı (herkese açık)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model yükle (nano ya da ihtiyacına göre değiştir)
model = YOLO("yolov8n.pt")
#model = YOLO(r"C:\Users\ACER\Desktop\human_detection\yolo\runs\detect\person_det4\weights\best.pt")


# Kutu çizimli görüntülerin kaydedileceği dizin
SAVE_DIR = "ornekler"
# Dizin yoksa oluştur
os.makedirs(SAVE_DIR, exist_ok=True)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Gelen resmi oku
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Tahmin yap
    results = model(img, conf=0.7)
    boxes = results[0].boxes.data.cpu().tolist()  # [x1,y1,x2,y2,conf,cls]

    # Bounding box çiz
    draw = ImageDraw.Draw(img)
    # Kalınlık: küçük kenarın %1'i kadar
    min_dim = min(img.width, img.height)
    line_width = max(1, int(min_dim * 0.01))
    for box in boxes:
        x1, y1, x2, y2, conf, cls = box
        draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=line_width)

    # Kaydedilecek dosya adı (zaman damgası)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"boxed_{timestamp}.png"
    save_path = os.path.join(SAVE_DIR, filename)
    # Resmi klasöre kaydet
    img.save(save_path, format="PNG")

    # Görüntüyü byte buffer'a kaydet
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    # Yanıta kutulu resmi PNG olarak dön
    return StreamingResponse(buf, media_type="image/png")
