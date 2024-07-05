from ultralytics import YOLO
import cv2
import cvzone
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
while True:
    success,img = cap.read()
    results = model.track(img, persist=True)
    for detection in results[0]:
        for box in detection.boxes:
            cls = int(box.cls[0])
            id = int(box.id[0])
            conf = int(box.conf[0] * 100)
            if cls == 0 and conf > 50:
                x1,y1,x2,y2 = box.xyxy[0]
                x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
                w,h = x2-x1, y2-y1
                cvzone.cornerRect(img, (x1, y1, w, h))
                cvzone.putTextRect(img,f"{classNames[cls]} {conf}%, ID: {id}".upper(), (x1 + 5, max(y1-15, 18)), thickness=2,scale=1)
    cv2.imshow('xx', img)
    if cv2.waitKey(1) == ord('q'):
        break
  