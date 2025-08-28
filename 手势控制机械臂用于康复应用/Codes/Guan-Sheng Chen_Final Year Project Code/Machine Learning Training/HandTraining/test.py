################################################################################
# Reference: https://www.computervision.zone/courses/hand-sign-detection-asl/
# Author: Guan-Sheng Chen
# Date: 9 May 2024
# Copyright, Guan-Sheng Chen 2024
################################################################################
# The file is to test the trained machine learning model from Teachable Machine
################################################################################
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
#Camera Setting
wCam, hCam = 1920, 1080
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
#package import
detector = HandDetector(maxHands=1)
classifier = Classifier("Model1/keras_model.h5", "Model1/labels.txt" )

offset = 20
imgSize = 300

folder = "Data/A"
counter = 0
pTime = 0

labels = ["HandDistance","Multi"]
index = 0
data = open('test.txt', 'w')

while True:
    success, img = cap.read()

    imgOutput = img.copy()
    hands, img = detector.findHands(img)    #with draw
    #hands = detector.findHands(img, draw=False)    #no draw, but the prediction will be fail
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255 # create a blank image
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset] # image cropping
        imgCropShape = imgCrop.shape
        aspectRatio = h / w
        # resize the image and obtain the prediction
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            #print(labels[index])
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        cv2.rectangle(imgOutput, (x - offset, y - offset - 50), (x - offset + 350, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x,y-26),cv2.FONT_HERSHEY_COMPLEX,1.7,(255,255,255),2)
        cv2.rectangle(imgOutput, (x-offset, y-offset), (x+w+offset, y+h+offset), (255, 0, 255),4)
        #cv2.imshow("ImageCrop", imgCrop)
        #cv2.imshow("ImageWhite", imgWhite)
        # Calculate FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(imgOutput, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
    print(fps)
    data.write(str(labels[index]) + '\n')
    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)
