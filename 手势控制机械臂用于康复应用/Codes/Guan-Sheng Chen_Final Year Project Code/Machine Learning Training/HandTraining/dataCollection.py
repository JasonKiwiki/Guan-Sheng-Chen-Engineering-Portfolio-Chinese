################################################################################
# Reference: https://www.computervision.zone/courses/hand-sign-detection-asl/
# Author: Guan-Sheng Chen
# Date: 9 May 2024
# Copyright, Guan-Sheng Chen 2024
################################################################################
# The file is to collect the image for Teachable Machine to do machine learning
################################################################################
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
#Camera Setting
wCam, hCam = 1920, 1080
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
#Hand Detector input
detector = HandDetector(maxHands=1)
offset = 20 #leave some space for cropped image
imgSize = 300
folder = "Data1/C"
counter = 0
pTime = 0 #FPS
while True:
    success, img = cap.read()
    hands, img = detector.findHands(img) #Hand Detection
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255 # create a blank image
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset] #image cropped
        imgCropShape = imgCrop.shape
        aspectRatio = h / w
        #image combination, resize the image and obtain the prediction
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    #press "s" to save the image
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
        print(counter)