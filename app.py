import cv2
import uvicorn
from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import numpy as np
import io
import json
from datetime import datetime

app = FastAPI(title="車両検査システム")

model = YOLO("yolov8n.pt")

def read_qr_code(image: np.ndarray) -> str:
    detector = cv2.QRCodeDetector()
    data, _, _ = detector.detectAndDecode(image)
    return data if data else None

def detect_damage(image: np.ndarray) -> list:
    results = model(image)
    damages = []
    for result in results:
        for box in result.boxes:
            damages.append({
                "class": result.names[int(box.cls)],
                "confidence": float(box.conf),
                "position": box.xyxy[0].tolist()
            })
    return damages

@app.post("/inspect")
async def inspect_vehicle(file: UploadFile = File(...)):
    contents = await file.read()
    image = cv2.imdecode(
        np.frombuffer(contents, np.uint8),
        cv2.IMREAD_COLOR
    )

    qr_data = read_qr_code(image)
    damages = detect_damage(image)

    result = {
        "vehicle_id": qr_data or "QR読み取り失敗",
        "inspection_time": datetime.now().isoformat(),
        "damage_detected": len(damages) > 0,
        "damage_count": len(damages),
        "damages": damages,
        "status": "要確認" if damages else "異常なし"
    }

    return result

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)