################################################################################
# Author: Guan-Sheng Chen
# Date: 9 May 2024
# Copyright, Guan-Sheng Chen 2024
################################################################################
# The file is the main control center of all tasks,
# and execute the initialization
################################################################################
import cv2
import HandTrackingModuleCombine as hem
import GestureRecognition as htt
import numpy as np
from cvzone.ClassificationModule import Classifier
import os
import subprocess
import time

#Camera Seeting
cap = cv2.VideoCapture(0)
cap.set(3, 1920)
cap.set(4, 1080)

#Package Imported
detector = hem.HandDetector(detectionCon=0.8, maxHands=2)
recognizer = htt.GestureRecognition(detectionCon=0.8, maxHands=2)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt" ) #machine learning model import
classifierL = Classifier("Model1/keras_model.h5", "Model1/labels.txt")#machine learning model import

write_video = True      #video saving
conMotion = 0           #identify the options
conMotionZero = True    #flag to identify the motion is defined or not
conMotionMul = 0        #identify the options (MultiHands)
conMotionZeroMul =True  #flag to identify the motion (MultiHands) is defined or not

test = 0
drawColor = (255, 0, 255)#virtual drawing brush initial color
xp, yp = 0, 0            #virtual drawing original point
reset = 0
con = 0
tt = 0
num = -2

folderR = "HandTracking/Right"#finger counting pirctures import
handList = os.listdir(folderR)#save in a list
data = open('FPS.txt', 'w')#reocrd data writing
pTime = 0#FPS


# Read the gesture image and store it in the list
print(handList)
overlayList = []
for imPath in handList:
    image = cv2.imread(f'{folderR}/{imPath}')
    overlayList.append(image)

#Print gesture image file list and number
print(len(overlayList))
for i in range(len(overlayList)):
    overlayList[i] = cv2.resize(overlayList[i], (200,200))


folderL = "HandTracking/Left"#finger counting pictures import (MultiHands)
handListL = os.listdir(folderL)

# Read the gesture image and store it in the list
print(handListL)
overlayListL = []
for imPathL in handListL:
    imageL = cv2.imread(f'{folderL}/{imPathL}')
    overlayListL.append(imageL)

#Print gesture image file list and number
print(len(overlayListL))
for i in range(len(overlayListL)):
    overlayListL[i] = cv2.resize(overlayListL[i], (200,200))
# video writer
if write_video:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('outputMain11.mp4', fourcc, 5.0, (1920, 1080))



while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)  #hand detection with Draw
    lmList, _ = detector.findPosition(img)  #enhance x and y position, and modify to receive lmList and _
    fingers = detector.fingersUp()          #finger detection

    # Separate the results for both hands
    imgOutput = img.copy()
    imgOutputR = img.copy()
    imgOutputL = img.copy()

#complex gesture recognition and finger classification
    for hand in hands:

        if len(hands) == 1:
            # Hand 1
            hand1 = hands[0]
            handType1 = hand1["type"]  #hand Type Left or Right
            #call trainModelRec for each hand (complex gesture recognition)
            if handType1 == "Right":#right hand
                conMotion, imgOutput, num = recognizer.gesture(imgOutput, detector, overlayList)#the function will return the motion is defined or not and some motions will through this function to execute the tasks
                if conMotion == 0 and conMotionZero:#if the motion is not defined
                    imgOutput = recognizer.trainModelRec(img, hand1, classifier)#complex gesture recognition
                    imgOutput = recognizer.classification(imgOutput, fingers)#finger classification

                else:
                    conMotionZero = False #motion is defined and next time it will go through executing the corresponding tasks from "gesture" function
                    test = conMotion
            elif handType1 == "Left":#left hand
                conMotion, imgOutput, num = recognizer.gesture(imgOutput, detector, overlayList)
                if conMotion == 0 and conMotionZero:
                    imgOutput = recognizer.trainModelRec(img, hand1, classifier)
                    imgOutput = recognizer.classification(imgOutput, fingers)

                else:
                    conMotionZero = False
                    test = conMotion
        #Two hands are detected (MultiHands), the motion should be also "MultiHands" to enter the recognition
        if len(hands) == 2 and conMotion == 2:
            hand1 = hands[0]
            fingers1 = detector.fingersUp()#finger detection
            handType1 = hand1["type"]  #hand Type Left or Right
            if handType1 == "Right":
                test = 0
                conMotionMul, imgOutput = recognizer.gestureL(imgOutput, detector, overlayList, overlayListL, hands, test)
                if conMotionMul == 0 and conMotionZeroMul:#motion is not defined (MultiHands)
                    hand2 = hands[1]    #save second hand
                    handType2 = hand2["type"]  #hand Type Left or Right
                    fingers2 = detector.fingersUp()#finger detection (second hand
                    imgOutputR = recognizer.trainModelRecR(img, hand1, classifier)#right hand complex gesture recognition (same as previous)
                    imgOutputL = recognizer.trainModelRecL(img, hand2, classifierL)#similar to complex gesture recognition, but the motions are changed to "HandDistance" or "FingerCounting(2)"
                    imgOutput = cv2.addWeighted(imgOutputR, 0.5, imgOutputL, 0.5, 0)#add both recognition frame and show on the screen
                else:
                    conMotionZero = False


            elif handType1 == "Left":

                conMotionMul, imgOutput = recognizer.gestureL(imgOutput, detector, overlayList, overlayListL, hands, test)
                if conMotionMul == 0 and conMotionZeroMul:
                    hand2 = hands[1]
                    handType2 = hand2["type"]  #hand Type Left or Right
                    fingers2 = detector.fingersUp()
                    imgOutputL = recognizer.trainModelRecL(img, hand1, classifierL)
                    imgOutputR = recognizer.trainModelRecR(img, hand2, classifier)
                    imgOutput = cv2.addWeighted(imgOutputR, 0.5, imgOutputL, 0.5, 0)
                else:
                    conMotionZero = False

    #Calculate FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(imgOutput, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN,3, (0, 255, 0), 3)
    tt = tt+1#used for dara record

    data.write(str(fps) + '\n')#used for dara record
    cv2.imshow("Combined Image", imgOutput)#show the frame
    #video output
    if write_video:
        out.write(imgOutput)
    if cv2.waitKey(5) & 0xFF == 27:#press ESC to stop and output the video

        if write_video:
            out.release()
        break
    cv2.waitKey(1)
    #if "RealTime" motion is defined, it will open other file to reduce the delay time and enhance its efficiency
    if conMotion == 4:
        cv2.destroyAllWindows()
        cap.release()
        subprocess.run(["python", "dobotRealTime.py"])




