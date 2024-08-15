import torch
from PIL import Image
import cv2
import os

# Load the trained YOLO model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/exp5/weights/best.pt')

def predict_and_show(image_path):
    # Load image
    image = Image.open(image_path)
    
    # Make prediction
    results = model(image)
    
    # Get bounding box coordinates and confidence
    bbox = results.xyxy[0].cpu().numpy()
    
    # Load image with OpenCV to draw bounding boxes
    img = cv2.imread(image_path)
    
    for x1, y1, x2, y2, conf, cls in bbox:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Show image with bounding boxes
    cv2.imshow('Predicted Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Define path to validation images
image_folder = '/Users/eddiehepburn/MSD-Classification/yolo/images/val'

# Predict and show results for all images in the validation set
for image_name in os.listdir(image_folder):
    if image_name.endswith(('.jpg', '.jpeg', '.png', '.webp')):
        image_path = os.path.join(image_folder, image_name)
        predict_and_show(image_path)
