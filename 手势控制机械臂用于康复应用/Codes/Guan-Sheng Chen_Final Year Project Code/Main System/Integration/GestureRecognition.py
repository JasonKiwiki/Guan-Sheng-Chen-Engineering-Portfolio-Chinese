################################################################################
# Reference: https://www.computervision.zone/courses/ai-virtual-painter/
# Reference: https://www.computervision.zone/courses/finger-counter/
# Reference: https://www.computervision.zone/courses/multiple-hand-gesture-control/
# Reference: https://www.computervision.zone/courses/hand-sign-detection-asl/
# Author: Guan-Sheng Chen
# Date: 9 May 2024
# Copyright, Guan-Sheng Chen 2024
################################################################################
# The file is to implement the tasks of complex gesture recognition,
# gesture "Single" (Finger Counting),
# gesture "MultiHands" (Hand Distance, FingerCounting(2)),
# gesture "Draw" (Virtual Drawing), Finger Classification,
# Connection of "DobotMovement"
################################################################################
import cv2
import mediapipe as mp
import math
import numpy as np
import subprocess
from cvzone.ClassificationModule import Classifier

# Initialization
class GestureRecognition():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, int(self.detectionCon), self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.index = 0  # Initialize index as an instance variable
        self.labels = ["Single", "MultiHands", "Draw", "RealTime"]
        self.labelsL = ["HandDistance", "FingerCounting(2)"]
        self.counts = {"Single": 0, "MultiHands": 0, "Draw": 0, "RealTime": 0}
        self.countsL = {"HandDistance": 0, "FingerCounting(2)": 0}
        self.temp = "Single"
        self.tempL = "A"
        self.offset = 20 # make sure the hole hand can be detected
        self.drawColor = (255, 0, 255)
        self.imgCanvas = np.zeros((1080, 1920, 3), np.uint8)
        self.xp = 0
        self.yp = 0
        self.shape = []
        self.classifier = Classifier("ModelShape/keras_model.h5", "ModelShape/labels.txt")
        self.labelShape = ["Circle", "X", "Line"]

    # Print which fingers are up
    def classification(self, img, fingers):
        finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

        up_fingers = [finger_names[i] for i, finger_state in enumerate(fingers) if finger_state == 1]

        if up_fingers:
            text = "Fingers up: " + ', '.join(up_fingers)
        else:
            text = "No fingers up"

        cv2.putText(img, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        return img

    #Complex Gesture recognition of Single, MultiHands, Draw, RealTime
    def trainModelRecR(self, img, hand1, classifier):
        imgSize = 300
        imgOutput = img.copy()
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255  # Default white image

        x, y, w, h = hand1['bbox']
        imgCrop = img[y - self.offset:y + h + self.offset, x - self.offset:x + w + self.offset] # image cropping
        if not imgCrop.size:
            # Handle the case where imgCrop is empty
            return imgOutput
        #imgCropShape = imgCrop.shape
        aspectRatio = h / w
        #Image resizing and combination
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            #imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            #imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
        # complex gesture recognition: self.index will identify which gesture is recognisied
        prediction, self.index = classifier.getPrediction(imgWhite, draw=False)

        cv2.rectangle(imgOutput, (x - self.offset, y - self.offset - 50), (x - self.offset + 250, y - self.offset - 50 + 50), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, self.labels[self.index], (x, y-26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x - self.offset, y - self.offset), (x + w + self.offset, y + h + self.offset), (255, 0, 255), 4)


        return imgOutput

    # Gesture recognition of HandDistance and FingerCounting(2)
    def trainModelRecL(self, img, hand1, classifierL):
        imgSize = 300
        imgOutput = img.copy()
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255  # Default white image


        x, y, w, h = hand1['bbox']
        imgCrop = img[y - self.offset:y + h + self.offset, x - self.offset:x + w + self.offset]
        if not imgCrop.size:
            # Handle the case where imgCrop is empty
            return imgOutput
        #imgCropShape = imgCrop.shape
        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            #imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            #imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        prediction, self.index = classifierL.getPrediction(imgWhite, draw=False)

        detectedLabelL = self.labelsL[self.index]  # detected label

        if self.tempL != detectedLabelL:
            self.counts[self.tempL] = 0
            print("Reset")

        if detectedLabelL in self.countsL:
            self.tempL = detectedLabelL
            self.countsL[detectedLabelL] += 1

        cv2.rectangle(imgOutput, (x - self.offset, y - self.offset - 50), (x - self.offset + 250, y - self.offset - 50 + 50),(255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, self.labelsL[self.index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x - self.offset, y - self.offset), (x + w + self.offset, y + h + self.offset), (255, 0, 255), 4)

        return imgOutput

    # Complex Gesture recognition of Single, MultiHands, Draw, RealTime (Only used for "MultiHands
    def trainModelRec(self, img, hand1, classifier):
        imgSize = 300
        imgOutput = img.copy()
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255  # Default white image

        x, y, w, h = hand1['bbox']
        imgCrop = img[y - self.offset:y + h + self.offset, x - self.offset:x + w + self.offset]
        if not imgCrop.size:
            # Handle the case where imgCrop is empty
            return imgOutput
        #imgCropShape = imgCrop.shape
        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            #imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            #imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        prediction, self.index = classifier.getPrediction(imgWhite, draw=False)
        detectedLabel = self.labels[self.index]  # detected label

        if self.temp != detectedLabel:
            self.counts[self.temp] = 0
            print("Reset")

        if detectedLabel in self.counts:
            self.temp = detectedLabel
            self.counts[detectedLabel] += 1

        cv2.rectangle(imgOutput, (x - self.offset, y - self.offset - 50),
                      (x - self.offset + 250, y - self.offset - 50 + 50), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, self.labels[self.index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x - self.offset, y - self.offset), (x + w + self.offset, y + h + self.offset),
                      (255, 0, 255), 4)

        return imgOutput
    # If one of gesture is recognised continuously 10 times, it will execute the corresponding tasks
    def gesture(self, imgee, detector, overlayList):
        con = 0
        num = -2
        if self.counts["Single"] > 10:
            print("Single")
            con = 1
            _, num = self.fingerCounting(imgee, detector, overlayList)
        elif self.counts["MultiHands"] > 10:
            print("MultiHands")
            con = 2
        elif self.counts["Draw"] > 10:
            print("Draw")
            con = 3
            imgee = self.virtualDraw(imgee, detector)
        elif self.counts["RealTime"] > 10:
            print("RealTime")
            con = 4

        return con, imgee, num

    # If one of gesture is recognised continuously 10 times, it will execute the corresponding tasks (used for "MultiHands")
    def gestureL(self, imgee, detector, overlayList, overlayListL, hands, test):
        con = 0
        if self.countsL["HandDistance"] > 10:
            print("HandDistance")
            con = 1

            imgee = self.handDistance(imgee, hands, detector)
        elif self.countsL["FingerCounting(2)"] > 10:
            print("FingerCounting(2)")
            con = 2
            imgee = self.fingerCountingMulti(imgee, detector, overlayList, overlayListL, hands, test)

        return con, imgee
    # Two Hand Distance Measurement
    def handDistance(self, img, hands, detector):
        if hands:
            # Hand 1
            hand1 = hands[0]
            centerPoint1 = hand1["center"]  # center of the hand cx,cy

            if len(hands) == 2:
                left=hands[0]['lmList']
                right=hands[1]['lmList']
                pointa=left[9] # left hand middle finger metacarpophalangeal joint
                pointb=right[9] #  right hand middle finger metacarpophalangeal joint
                # it is used for verification in Final System Testing to calculate its calculation accuracy
                dis = math.sqrt((pointa[0] - pointb[0])**2 + (pointa[1] - pointb[1])**2 + (pointa[2] - pointb[2])**2)
                hand2 = hands[1]
                centerPoint2 = hand2["center"]  # center of the hand cx,cy
                # distance calculation
                length, info, img = detector.findDistance(centerPoint1, centerPoint2, img)  # with draw
                print(length, dis)


                # Extract the center point of the line
                _, _, _, _, cx, cy = info

                # Adjust the text position to be on the line's center point
                cv2.putText(img, f"Distance: {int(length)} pixels", (cx-200, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),2)
                # Print the length on the screen
                cv2.putText(img, f"Distance: {int(length)} pixels", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),2)
        return img

    # "Singer" Gesture - Finger Counting
    def fingerCounting(self, imgee, detector, overlayList):

        fingercount = detector.fingersUp()
        # Count the total number of fingers
        totalFingers = fingercount.count(1)
        print(totalFingers)
        # using Finger Detection returned array to identify which fingers are up
        if fingercount == [0, 0, 0, 0, 0]:# 0
            # Display the icon of the corresponding gesture on the image
            h, w, c = overlayList[0].shape
            imgee[0:h, 0:w] = overlayList[0]
            a = 0
        elif fingercount == [0, 1, 0, 0, 0]:# 1
            h, w, c = overlayList[1].shape
            imgee[0:h, 0:w] = overlayList[1]
            a = 1
        elif fingercount == [0, 1, 1, 0, 0]:# 2
            h, w, c = overlayList[2].shape
            imgee[0:h, 0:w] = overlayList[2]
            a = 2
        elif fingercount == [0, 1, 1, 1, 0]:# 3
            h, w, c = overlayList[3].shape
            imgee[0:h, 0:w] = overlayList[3]
            a = 3
        elif fingercount == [0, 1, 1, 1, 1]:# 4
            h, w, c = overlayList[4].shape
            imgee[0:h, 0:w] = overlayList[4]
            a = 4
        elif fingercount == [1, 1, 1, 1, 1]:# 5
            h, w, c = overlayList[5].shape
            imgee[0:h, 0:w] = overlayList[5]
            a = 5
        elif fingercount == [1, 0, 0, 0, 1]:# 6
            h, w, c = overlayList[6].shape
            imgee[0:h, 0:w] = overlayList[6]
            totalFingers = 6 # it will show the gesture is number 6
            a = 6
        elif fingercount == [1, 1, 0, 0, 0]:# 7
            h, w, c = overlayList[7].shape
            imgee[0:h, 0:w] = overlayList[7]
            totalFingers = 7 # it will show the gesture is number 7
            a = 7
        elif fingercount == [1, 1, 1, 0, 0]:# 8
            h, w, c = overlayList[8].shape
            imgee[0:h, 0:w] = overlayList[8]
            totalFingers = 8 # it will show the gesture is number 8
            a = 8
        elif fingercount == [1, 1, 1, 1, 0]:# 9
            h, w, c = overlayList[9].shape
            imgee[0:h, 0:w] = overlayList[9]
            totalFingers = 9 # it will show the gesture is number 9
            a = 9
        else:
            imgee = self.classification(imgee, fingercount) # the gesture is not in the list of 0 to 9
            a = -1

        # Draw finger-number rectangles and text on the image
        cv2.rectangle(imgee, (20, 225), (170, 425), (255, 0, 0), cv2.FILLED)
        cv2.putText(imgee, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN, 10, (255, 255, 255), 25)

        return imgee, a


    # "MultiHands" Gesture tasks - FingerCounting(2)
    def fingerCountingMulti(self, imgee, detector, overlayList, overlayListL, hands, test):
        print(test)
        totalFingersR = 0
        totalFingersL = 0
        if test == 0: # "test" is used to identify which hands (left or right) is detected first
            print("0")
            fingersR = detector.fingersUpMulti(hands[0])
            fingersL = detector.fingersUpMulti(hands[1])
            # Count the total number of fingers
            totalFingersR = fingersR.count(1)
            totalFingersL = fingersL.count(1)
            # using Finger Detection returned array to identify which fingers are up
            if fingersR == [0, 0, 0, 0, 0]:# 0
                # Display the icon of the corresponding gesture on the image
                h, w, c = overlayList[0].shape
                imgee[0:h, 0:w] = overlayList[0]
            elif fingersR == [0, 1, 0, 0, 0]:# 1
                h, w, c = overlayList[1].shape
                imgee[0:h, 0:w] = overlayList[1]
            elif fingersR == [0, 1, 1, 0, 0]:# 2
                h, w, c = overlayList[2].shape
                imgee[0:h, 0:w] = overlayList[2]
            elif fingersR == [0, 1, 1, 1, 0]:# 3
                h, w, c = overlayList[3].shape
                imgee[0:h, 0:w] = overlayList[3]
            elif fingersR == [0, 1, 1, 1, 1]:# 4
                h, w, c = overlayList[4].shape
                imgee[0:h, 0:w] = overlayList[4]
            elif fingersR == [1, 1, 1, 1, 1]:# 5
                h, w, c = overlayList[5].shape
                imgee[0:h, 0:w] = overlayList[5]
            elif fingersR == [1, 0, 0, 0, 1]:# 6
                h, w, c = overlayList[6].shape
                imgee[0:h, 0:w] = overlayList[6]
                totalFingersR = 6 # it will show the gesture is number 6
            elif fingersR == [1, 1, 0, 0, 0]:# 7
                h, w, c = overlayList[7].shape
                imgee[0:h, 0:w] = overlayList[7]
                totalFingersR = 7 # it will show the gesture is number 7
            elif fingersR == [1, 1, 1, 0, 0]:# 8
                h, w, c = overlayList[8].shape
                imgee[0:h, 0:w] = overlayList[8]
                totalFingersR = 8 # it will show the gesture is number 8
            elif fingersR == [1, 1, 1, 1, 0]:# 9
                h, w, c = overlayList[9].shape
                imgee[0:h, 0:w] = overlayList[9]
                totalFingersR = 9 # it will show the gesture is number 9
            else:
                imgee = self.classification(imgee, fingersR) # the gesture is not in the list of 0 to 9

            if fingersL == [0, 0, 0, 0, 0]:
                # Display the icon of the corresponding gesture on the image
                h, w, c = overlayListL[0].shape
                imgee[0:h, 200:200+w] = overlayListL[0]
            elif fingersL == [0, 1, 0, 0, 0]:
                h, w, c = overlayListL[1].shape
                imgee[0:h, 200:200+w] = overlayListL[1]
            elif fingersL == [0, 1, 1, 0, 0]:
                h, w, c = overlayListL[2].shape
                imgee[0:h, 200:200+w] = overlayListL[2]
            elif fingersL == [0, 1, 1, 1, 0]:
                h, w, c = overlayListL[3].shape
                imgee[0:h, 200:200+w] = overlayListL[3]
            elif fingersL == [0, 1, 1, 1, 1]:
                h, w, c = overlayListL[4].shape
                imgee[0:h, 200:200+w] = overlayListL[4]
            elif fingersL == [1, 1, 1, 1, 1]:
                h, w, c = overlayListL[5].shape
                imgee[0:h, 200:200+w] = overlayListL[5]
            elif fingersL == [1, 0, 0, 0, 1]:
                h, w, c = overlayListL[6].shape
                imgee[0:h, 200:200+w] = overlayListL[6]
                totalFingersL = 6
            elif fingersL == [1, 1, 0, 0, 0]:
                h, w, c = overlayListL[7].shape
                imgee[0:h, 200:200+w] = overlayListL[7]
                totalFingersL = 7
            elif fingersL == [1, 1, 1, 0, 0]:
                h, w, c = overlayListL[8].shape
                imgee[0:h, 200:200+w] = overlayListL[8]
                totalFingersL = 8
            elif fingersL == [1, 1, 1, 1, 0]:
                h, w, c = overlayListL[9].shape
                imgee[0:h, 200:200+w] = overlayListL[9]
                totalFingersL = 9
            else:
                imgee = self.classification(imgee, fingersL)

        elif test == 1:
            print("1")
            fingersL = detector.fingersUpMulti(hands[0])
            fingersR = detector.fingersUpMulti(hands[1])
            # Count the total number of fingers
            totalFingersR = fingersR.count(1)
            totalFingersL = fingersL.count(1)

            # Check if only the index finger is up
            if fingersR == [0, 0, 0, 0, 0]:
                # Display the icon of the corresponding gesture on the image
                h, w, c = overlayList[0].shape
                imgee[0:h, 0:w] = overlayList[0]
            elif fingersR == [0, 1, 0, 0, 0]:
                h, w, c = overlayList[1].shape
                imgee[0:h, 0:w] = overlayList[1]
            elif fingersR == [0, 1, 1, 0, 0]:
                h, w, c = overlayList[2].shape
                imgee[0:h, 0:w] = overlayList[2]
            elif fingersR == [0, 1, 1, 1, 0]:
                h, w, c = overlayList[3].shape
                imgee[0:h, 0:w] = overlayList[3]
            elif fingersR == [0, 1, 1, 1, 1]:
                h, w, c = overlayList[4].shape
                imgee[0:h, 0:w] = overlayList[4]
            elif fingersR == [1, 1, 1, 1, 1]:
                h, w, c = overlayList[5].shape
                imgee[0:h, 0:w] = overlayList[5]
            elif fingersR == [1, 0, 0, 0, 1]:
                h, w, c = overlayList[6].shape
                imgee[0:h, 0:w] = overlayList[6]
                totalFingersR = 6
            elif fingersR == [1, 1, 0, 0, 0]:
                h, w, c = overlayList[7].shape
                imgee[0:h, 0:w] = overlayList[7]
                totalFingersR = 7
            elif fingersR == [1, 1, 1, 0, 0]:
                h, w, c = overlayList[8].shape
                imgee[0:h, 0:w] = overlayList[8]
                totalFingersR = 8
            elif fingersR == [1, 1, 1, 1, 0]:
                h, w, c = overlayList[9].shape
                imgee[0:h, 0:w] = overlayList[9]
                totalFingersR = 9
            else:
                imgee = self.classification(imgee, fingersR)

            if fingersL == [0, 0, 0, 0, 0]:
                # Display the icon of the corresponding gesture on the image
                h, w, c = overlayListL[0].shape
                imgee[0:h, 200:200 + w] = overlayListL[0]
            elif fingersL == [0, 1, 0, 0, 0]:
                h, w, c = overlayListL[1].shape
                imgee[0:h, 200:200 + w] = overlayListL[1]
            elif fingersL == [0, 1, 1, 0, 0]:
                h, w, c = overlayListL[2].shape
                imgee[0:h, 200:200 + w] = overlayListL[2]
            elif fingersL == [0, 1, 1, 1, 0]:
                h, w, c = overlayListL[3].shape
                imgee[0:h, 200:200 + w] = overlayListL[3]
            elif fingersL == [0, 1, 1, 1, 1]:
                h, w, c = overlayListL[4].shape
                imgee[0:h, 200:200 + w] = overlayListL[4]
            elif fingersL == [1, 1, 1, 1, 1]:
                h, w, c = overlayListL[5].shape
                imgee[0:h, 200:200 + w] = overlayListL[5]
            elif fingersL == [1, 0, 0, 0, 1]:
                h, w, c = overlayListL[6].shape
                imgee[0:h, 200:200 + w] = overlayListL[6]
                totalFingersL = 6
            elif fingersL == [1, 1, 0, 0, 0]:
                h, w, c = overlayListL[7].shape
                imgee[0:h, 200:200 + w] = overlayListL[7]
                totalFingersL = 7
            elif fingersL == [1, 1, 1, 0, 0]:
                h, w, c = overlayListL[8].shape
                imgee[0:h, 200:200 + w] = overlayListL[8]
                totalFingersL = 8
            elif fingersL == [1, 1, 1, 1, 0]:
                h, w, c = overlayListL[9].shape
                imgee[0:h, 200:200 + w] = overlayListL[9]
                totalFingersL = 9
            else:
                imgee = self.classification(imgee, fingersL)

        # Draw finger-number rectangles and text on the image
        cv2.rectangle(imgee, (20, 225), (170, 425), (255, 0, 0), cv2.FILLED)
        cv2.putText(imgee, str(totalFingersR), (45, 375), cv2.FONT_HERSHEY_PLAIN, 10, (255, 255, 255), 25)
        cv2.rectangle(imgee, (170, 225), (320, 425), (255, 0, 0), cv2.FILLED)
        cv2.putText(imgee, str(totalFingersL), (195, 375), cv2.FONT_HERSHEY_PLAIN, 10, (255, 255, 255), 25)

        return imgee

    # "Draw" Gesture Tasks - Virtual Drawing
    def virtualDraw(self, imgee, detector):

        guessShape = False #flag to identify the shape on the screen
        brushThickness = 25
        eraserThickness = 100

        symbol = [0,0]#used for save the shape and its length
        # Find hand landmarks
        lmList, _ = detector.findPosition(imgee, draw=False)


        if lmList:  # Check if lmList is not empty

            # tip of index fingers
            x1, y1 = lmList[8][1:]
            # tip of middle fingers
            x2, y2 = lmList[12][1:]

            # Check which fingers are up
            fingers = detector.fingersUp()

            # Selection mode - two fingers are up
            if fingers[1] and fingers[2]:
                self.xp, self.yp = 0, 0
                if fingers[4]:
                    self.drawColor = (255, 0, 255) # brush color pink
                elif fingers[3]:
                    self.drawColor = (0, 0, 255) # brush color red
                elif fingers[0]:
                    self.drawColor = (0, 0, 0) # brush color black (Erase Mode)
                cv2.rectangle(imgee, (x1, y1 - 25), (x2, y2 + 25), self.drawColor, cv2.FILLED)

            # If drawing mode - index finger is up
            if fingers[1] and fingers[2] == False:
                cv2.circle(imgee, (x1, y1), 15, self.drawColor, cv2.FILLED)
                #Renew the original point dynamically
                if self.xp == 0 and self.yp == 0:
                    self.xp, self.yp = x1, y1

                # Erase Mode
                if self.drawColor == (0, 0, 0):
                    cv2.line(imgee, (self.xp, self.yp), (x1, y1), self.drawColor, eraserThickness)
                    cv2.line(self.imgCanvas, (self.xp, self.yp), (x1, y1), self.drawColor, eraserThickness)

                # Store the coordinates of the drawn line
                shapeCoords = [(self.xp, self.yp), (x1, y1)]
                self.shape.append(shapeCoords)

                cv2.line(imgee, (self.xp, self.yp), (x1, y1), self.drawColor, brushThickness)
                cv2.line(self.imgCanvas, (self.xp, self.yp), (x1, y1), self.drawColor, brushThickness)

                self.xp, self.yp = x1, y1

        # Alternative method to add two images or use weight
        imgGray = cv2.cvtColor(self.imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        imgee = cv2.bitwise_and(imgee, imgInv)
        imgee = cv2.bitwise_or(imgee, self.imgCanvas)

        #Fingers to confirm to distinguish the shape
        fingerCon = detector.fingersUp()

        if fingerCon == [1,1,1,1,1]:#sign to identify the shape on the screen
            guessShape = True
        # If guessShape flag is set
        if guessShape:

            # Convert the canvas to grayscale
            imgGray = cv2.cvtColor(self.imgCanvas, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY)



            # Find contours in the binary image
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Machine learning is used to identify the shape on the frame
            prediction, indexShape = self.classifier.getPrediction(self.imgCanvas, draw=False)
            print(self.labelShape[indexShape])
            for contour in contours:
                # Circle
                if self.labelShape[indexShape] == "Circle":
                    # Calculate the minimum enclosing circle
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    center = (int(x), int(y))
                    radius = int(radius)

                    # Draw the circle on the image
                    cv2.circle(imgee, center, radius, (0, 255, 0), 2)

                    cv2.putText(imgee, "Circle", (1920 - 960, 1080 - 540), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                    print("Circle with radius:", radius)
                    symbol[0] = 2
                    symbol[1] = radius  # length
                    guessSuccess = True
                    self.dobotControl(symbol, guessSuccess)
                # Symbol "X"
                elif self.labelShape[indexShape] == "X":
                    # Draw bounding box around the "X" symbol
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(imgee, (x, y), (x + w, y + h), (255, 0, 0), 2)

                    # Calculate the length of one side of the "X" symbol
                    length_of_side = w / 2  # Assuming "X" is symmetric

                    # Draw the "X" symbol on the screen
                    x_center = x + w // 2
                    y_center = y + h // 2

                    # Draw the lines of the "X" symbol based on the calculated length
                    cv2.line(imgee, (x_center - int(length_of_side), y_center - int(length_of_side)),(x_center + int(length_of_side), y_center + int(length_of_side)), (0, 255, 0), 2)
                    cv2.line(imgee, (x_center - int(length_of_side), y_center + int(length_of_side)),(x_center + int(length_of_side), y_center - int(length_of_side)), (0, 255, 0), 2)
                    print("X WITH SIDE:", length_of_side)
                    symbol[0] = 3
                    symbol[1] = length_of_side  # length
                    # Print the length on the screen
                    cv2.putText(imgee, f"Length of side: {length_of_side}", (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    guessSuccess = True
                    self.dobotControl(symbol, guessSuccess)
                # Line
                elif self.labelShape[indexShape] == "Line":
                    # Two vertices, likely a line
                    cv2.putText(imgee, "Line", (1920 - 200, 1080 - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                    print("Line")

                    # Draw a rectangle around the detected line
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(imgee, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Calculate the length of the line
                    line_length = int((w ** 2 + h ** 2) ** 0.5)

                    symbol[0] = 1
                    symbol[1] = line_length  # length

                    # Print the length on the screen
                    cv2.putText(imgee, f"Length of line: {line_length}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 0, 0), 2)

                    # Draw a dot at the center of the line
                    cv2.circle(imgee, (x + w // 2, y + h // 2), 5, (255, 0, 0), cv2.FILLED)

                    # Draw a perfect line based on its length
                    angle_rad = math.atan2(h, w)
                    line_endpoint_x = int(x + line_length * math.cos(angle_rad))
                    line_endpoint_y = int(y + line_length * math.sin(angle_rad))
                    cv2.line(imgee, (x + w // 2, y + h // 2), (line_endpoint_x, line_endpoint_y), (0, 0, 255), 2)
                    guessSuccess = True
                    self.dobotControl(symbol, guessSuccess)

            #Reset the canvas and the guess flag
            self.imgCanvas = np.zeros((1080, 1920, 3), np.uint8)
            #guessShape = False
            print(symbol)

        return imgee

    #Send the length and the type of pattern to DobotMovement
    def dobotControl(self, symbol, guessSuccess):
        if symbol[0] == 2 and guessSuccess:
            subprocess.run(["python", "DobotMovement.py", str(symbol[0]), str(symbol[1])])
        elif symbol[0] == 3 and guessSuccess:
            subprocess.run(["python", "DobotMovement.py", str(symbol[0]), str(symbol[1])])
        elif symbol[0] == 1 and guessSuccess:
            subprocess.run(["python", "DobotMovement.py", str(symbol[0]), str(symbol[1])])
