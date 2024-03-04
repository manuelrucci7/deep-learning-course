from fastapi import FastAPI, UploadFile, File
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import numpy as np
from ultralytics import YOLO
import cv2
from contextlib import asynccontextmanager
import json


# Download the image
# curl -O "http://.com/path_to_your_image.jpg"
# Send the imageexample

# curl -X POST "http://localhost:8000/predict" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@/home/t3lab/Desktop/courses/images/2024-02-19_19-38-26.png"
# pip install fastapi uvicorn gunicorn ultralytics opencv-python  python-multipart
# gunicorn -w 4 -k uvicorn.workers.UvicornWorker app_server:app
# uvicorn alarm_server:app --reload
# Lifespan: https://fastapi.tiangolo.com/advanced/events/
# uvicorn  app_server:app --host 0.0.0.0 --port 8000

model = None

origins = [
    "http://localhost:8000",  # Allow requests from this origin
    # Add more origins if needed
]


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def load_model():
    global model
    model = YOLO("yolov8n.pt")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    input_data = np.array(image)

    #input_data = Image.open("2024-02-19_19-38-26.png")
    res = model.predict(input_data, imgsz=640, conf=0.4, iou=0.1, verbose=False)

    # Check if there are any detections
    names_list = []
    score_list = []
    boxes_list = []
    class_id_list = []

    if res[0].boxes is not None:
    
        im_draw = input_data.copy()
        names = res[0].names
        scores = res[0].boxes.conf.cpu().numpy()
        class_ids = res[0].boxes.cls.cpu().numpy()
        boxes = res[0].boxes.xyxy.cpu().numpy()

        for i in range(0, len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            score = float(scores[i])
            class_id = int(class_ids[i])
            #cv2.rectangle(im_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #cv2.putText(im_draw, f"{names[class_id]}: {score:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            names_list.append(names[class_id])
            score_list.append(score)
            boxes_list.append([x1, y1, x2, y2])
            class_id_list.append(class_id)

        #cv2.imshow("Image", im_draw)
        #cv2.waitKey(0)
    
    res_dict = {
        "names": names_list,
        "scores": score_list,
        "boxes": boxes_list,
        "class_ids": class_id_list
    }
    return {"prediction": res_dict}



