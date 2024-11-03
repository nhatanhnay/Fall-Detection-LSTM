import torch
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
# Tải mô hình YOLOv8 cho pose
model = YOLO('yolov8x-pose.pt')

# Chạy inference trên một bức ảnh (sử dụng GPU nếu có)
image_path = r'data\fall-01-cam0-rgb\fall-01-cam0-rgb-001.png'
device = '0' if torch.cuda.is_available() else 'cpu'
results = model(image_path, device=device)
# Trích xuất keypoints
keypoints = results[0].keypoints
if keypoints is not None:

    # Tải ảnh để trực quan hóa
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Chuyển đổi tensor keypoints sang numpy
    keypoints_np = keypoints.cpu().numpy() if torch.cuda.is_available() else keypoints.numpy()

    # Trực quan hóa keypoints trên ảnh
    data = np.concatenate((np.expand_dims(keypoints_np.conf, axis=-1),keypoints_np.xyn),axis=2)
    data_raw = keypoints_np.data
    if len(data[0]) > 1:
        data = data[0]
    for i in range(17):
        img_rgb = cv2.circle(img_rgb, (int(data_raw[0][i][0]), int(data_raw[0][i][1])), 2, (0, 255, 0), -1)
    print(data)
    cv2.imshow('image', img_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print("No keypoints detected.")
