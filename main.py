from ultralytics import YOLO
import cv2
import cvzone
from sort import *
model = YOLO('../Yolo-Weights/yolov8n.pt')
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

tracker = Sort(20,)
dtetctions = np.empty((0,5))
while True:
    success,img = cap.read()
    results = model(img, stream=True)
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = int(box.conf[0] * 100)
            if cls == 0 and conf > 50:
                x1,y1,x2,y2 = box.xyxy[0]
                x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
                w,h = x2-x1, y2-y1
                dtetction = np.array([x1,y1,x2,y2,conf])
                dtetctions = np.vstack((dtetctions, dtetction))
                cvzone.cornerRect(img, (x1, y1, w, h))
                cvzone.putTextRect(img,f"{classNames[cls]} {conf}%".upper(), (x1 + 5, max(y1-15, 18)), thickness=2,scale=1)
    results = tracker.update(dtetctions)
    for result in results:
        print(result)
    cv2.imshow('xx', img)
    if cv2.waitKey(1) == ord('q'):
        break