# import data manipulation libraries
import pandas as pd
import torch
import numpy as np
# import other libraries
import os
import warnings
# import computer vision libraries
import cv2
import mediapipe as mp
warnings.filterwarnings('ignore')

import time
import os

import torch
import torchvision
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision.utils import make_grid
from torchvision import transforms as T
from torchvision import models, datasets
from torchvision.models import resnet50


import numpy as np
from matplotlib import pyplot as plt
from random import randint
import pandas as pd


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(64 * 7 * 7, 1024)  
        self.fc2 = nn.Linear(1024, 25) 

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  
        x = self.fc2(x)
        return x
    
# Create an instance of your custom CNN model
cnn = CNN()

# Load the model's state dictionary
model_path = '/Users/faithqiao/Downloads/dss-decal/model.pt'
cnn.load_state_dict(torch.load(model_path))

# Set the model to evaluation mode
cnn.eval()

# use the webcam and the model to predict the sign language letter
# initialize the webcam
cap = cv2.VideoCapture(0)
# set the font
font = cv2.FONT_HERSHEY_SIMPLEX
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

mp_drawing = mp.solutions.drawing_utils

labels = {i-1: chr(i+64) for i in range(1, 27)}

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_resized = cv2.resize(gray, (28, 28))
    gray_resized = gray_resized.reshape(1, 28, 28, 1) / 255.0
    
    return gray_resized

cap = cv2.VideoCapture(0)

def get_prediction(image):
    preprocessed_image = preprocess_image(image)
    input_tensor = torch.from_numpy(preprocessed_image).float()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        output = cnn(input_tensor)
    output_np = output.cpu().numpy()
    return np.argsort(output_np)[0][-3:]

# def get_prediction(image):
#     preprocessed_image = preprocess_image(image)
#     prediction = cnn.predict(preprocessed_image)
#     # return 3 largest probabilities
#     return np.argsort(prediction)[0][-3:]

def segment_hand(frame):
    # Convert the image from BGR to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define a range for skin color values in HSV space
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    # Threshold the HSV image to get only skin colors
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If any contour is found
    if contours:
        # Get the largest contour based on area
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get the bounding rectangle around the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Return the segment of the frame that contains the hand
        return frame[y:y+h, x:x+w]
    else:
        return None

def get_hand_roi(frame, landmarks):
    height, width, _ = frame.shape
    
    # Convert relative landmarks coordinates to absolute coordinates
    landmarks_abs = []
    for landmark in landmarks:
        landmarks_abs.append((int(landmark[0] * width), int(landmark[1] * height)))
    
    # Get coordinates of the bounding box
    x_coordinates = [coordinate[0] for coordinate in landmarks_abs]
    y_coordinates = [coordinate[1] for coordinate in landmarks_abs]
    
    x_min, x_max = min(x_coordinates), max(x_coordinates)
    y_min, y_max = min(y_coordinates), max(y_coordinates)
    
    # Adding some padding to the bounding box for better capture of the hand
    padding = 100
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(width, x_max + padding)
    y_max = min(height, y_max + padding)
    
    # Draw a rectangle around the detected hand
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    
    # Crop the hand ROI
    hand_roi = frame[y_min:y_max, x_min:x_max]
    
    return hand_roi


cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    
    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame and get the hand landmarks
    results = hands.process(image_rgb)
    
    # If hand landmarks are found, process and display them
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # Drawing hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get the coordinates of the hand landmarks
            landmark_list = []
            for landmark in landmarks.landmark:
                landmark_list.append([landmark.x, landmark.y, landmark.z])
            
            # Your code to use the landmarks for cropping and prediction can go here
            
            # For demonstration purposes, let's assume you have a function `get_hand_roi`
            # that crops the hand area based on landmarks and returns it
            hand_roi = get_hand_roi(frame, landmark_list)
            if hand_roi.size > 0:
                prediction = get_prediction(hand_roi)
                cv2.putText(frame, f'Prediction: {[labels.get(pred) for pred in prediction]}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                
    cv2.imshow('Hand Tracking', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()