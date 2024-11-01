import cv2
import numpy as np
from tensorflow.keras.models import load_model
from ultralytics import YOLO 
import torch


model = load_model('model.h5')
detect_keypoints = YOLO('yolov8x-pose.pt')
# Load the camera
# cap = cv2.VideoCapture(0)
# Load the video
cap = cv2.VideoCapture('vid3.mp4')

n_timesteps = 15  
batch_x = np.empty((15, 17, 3), float) 
fall_detected = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # End of the video
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    keypoint = detect_keypoints(img_rgb,device='0',verbose=False)
    keypoints = keypoint[0].keypoints

    for keypoint in keypoints:
        if keypoints.conf is None:
            data = np.zeros((1,17,3))
        else:
            keypoints = keypoints.cpu().numpy() if torch.cuda.is_available() else keypoints.numpy()
            data = np.concatenate((np.expand_dims(keypoints.conf, axis=-1),keypoints.xyn),axis=2)
        batch_x = np.append(batch_x[1:], data, axis=0)
        prediction = model.predict(np.expand_dims(batch_x, axis=0),verbose=0)
        fall_class = np.argmax(prediction, axis=1)
        cv2.putText(img_rgb, 'FALL' if fall_class == 1 else 'NO FALL', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
# show the video
    cv2.imshow('video', img_rgb)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
