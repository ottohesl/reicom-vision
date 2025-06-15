import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageDraw, ImageFont

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("no")
        break

    cv2.imshow('yolo', frame)
    key = cv2.waitKey(1)
    if key != -1:
        break
cap.release()
