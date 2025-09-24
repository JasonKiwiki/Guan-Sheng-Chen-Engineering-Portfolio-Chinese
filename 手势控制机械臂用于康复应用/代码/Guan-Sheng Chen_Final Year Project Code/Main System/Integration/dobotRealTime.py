################################################################################
# Reference: https://github.com/Crazycurly/gesture_MeArm
# Reference: https://www.computervision.zone/courses/finger-counter/
# Reference: https://www.dobot-robots.com/service/download-center
# Aurthor: Guan-Sheng Chen
# Date: 9 May 2024
# Copyright, Guan-Sheng Chen 2024
################################################################################
# The file is to implement gesture "RealTime" task with normal movement,
# a gripper and a suction cup
################################################################################
import cv2
import mediapipe as mp
import DobotDllType as dType
import HandTrackingModuleCombine as htt
import pandas as pd
import time

# Dobot Connection
CON_STR = {
    dType.DobotConnect.DobotConnect_NoError: "DobotConnect_NoError",
    dType.DobotConnect.DobotConnect_NotFound: "DobotConnect_NotFound",
    dType.DobotConnect.DobotConnect_Occupied: "DobotConnect_Occupied"
}
api = dType.load()

# Initialize Dobot
state = dType.ConnectDobot(api, "", 115200)[0]
print("Connect status:", CON_STR[state])
# Data record
write_video = True
data = open('FPS.txt', 'w')
data1 = open('exe.txt', 'w')
data2 = open('exe1.txt', 'w')
pTime = 0 #FPS
detector = htt.HandDetector(detectionCon=0.8, maxHands=2)   #package import
#Base angle (Joint Angle 1) Initialization
minX = 90
mediumX = 0
maxX = -90
# Utilize angle between wrist and index finger to control Joint angle 1
anglePalmMin = -50
anglePalmMedium = 50
# Rear arm angle (Joint Angle 2) Initialization
minY = 80
mediumY = 40
maxY = 0
# Utilize wrist y to control y axis
minYWrist = 0.3
maxYWrist = 0.7
# Fore arm (Joint Angle 4) Initialization
minZ = 80
mediumZ = 40
maxZ = 0
# use palm size to control z axis
minPalmSize = 0.1
maxPalmSize = 0.3
pas = 0
# End-effector rotation (Joint Angle 4) Initialization
rotateMin = 150
rotateMid = 75
rotateMax = 0

jointAngle = [mediumX, mediumY, mediumZ] # [joint 1, joint 2, joint 3]
jointAnglePrev = jointAngle

# mediapipe hand initialization
mpDrawing = mp.solutions.drawing_utils
mpDrawingStyles = mp.solutions.drawing_styles
mpHands = mp.solutions.hands

#Camera setting
cap = cv2.VideoCapture(0)
cap.set(3, 1920)
cap.set(4, 1080)

# Open a file in append mode (Data Recording)
data_filewrist = open('Record/wrist.txt', 'w')
data_filefangle = open('Record/fingercoodinate.txt', 'w')
data_fileindex = open('Record/index.txt', 'w')
data_filej1 = open('Record/joint1.txt', 'w')
data_filej2 = open('Record/joint2.txt', 'w')
data_filej3 = open('Record/joint3.txt', 'w')
data_filej4 = open('Record/joint4.txt', 'w')
data_filex = open('Record/x.txt', 'w')
data_filey = open('Record/y.txt', 'w')
data_filez = open('Record/z.txt', 'w')
data_filer = open('Record/r.txt', 'w')
data_filerotate = open('Record/rotate.txt', 'w')

# video writer
if write_video:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.mp4', fourcc, 5.0, (1920, 1080))

#Two function to project angle to the new range
limitRange = lambda n, minn, maxn: max(min(maxn, n), minn)
mapRange = lambda x, in_min, in_max, out_min, out_max: int((x - in_min) * (out_max - out_min) // (in_max - in_min) + out_min)

#Joint Angle 4 Calculation
def rotatAngle(hand_landmarks):
    wrist = hand_landmarks.landmark[0]
    indexFingerMCP = hand_landmarks.landmark[5]
    # calculate the distance between the wrist and the index finger
    palmSize = ((wrist.x - indexFingerMCP.x) ** 2 + (wrist.y - indexFingerMCP.y) ** 2 + (
                wrist.z - indexFingerMCP.z) ** 2) ** 0.5
    # calculate joint angle 4
    distance = palmSize
    angle = (wrist.x - indexFingerMCP.x) / distance  # calculate the radian between the wrist and the index finger
    angle = int(angle * 180 / 3.1415926)  # convert radian to degree
    angle1 = angle
    angle = limitRange(angle, anglePalmMin, anglePalmMedium)
    angle2 = angle
    rotateAngle = mapRange(angle, anglePalmMin, anglePalmMedium, rotateMax, rotateMin)
    #Path to your Excel file
    file_path = 'Record/rotate.xlsx'
    # Creating a single DataFrame with all required columns
    df = pd.DataFrame(columns=[
        'x(wrist)', 'y(wrist)', 'z(wrist)',
        'x(index)', 'y(index)', 'z(index)',
        'palm',
        'angle1', 'angle2', 'joint4(sys)'
    ])
    # Create a new DataFrame for the current iteration's data
    new_row = pd.DataFrame({
        'x(wrist)': [wrist.x], 'y(wrist)': [wrist.y], 'z(wrist)': [wrist.z],
        'x(index)': [indexFingerMCP.x], 'y(index)': [indexFingerMCP.y], 'z(index)': [indexFingerMCP.z],
        'palm': [palmSize],
        'angle1': [angle1], 'angle2': [angle2],
        'joint4(sys)': [rotateAngle]
    })
    # Use concat to add the new row to the existing DataFrame
    df = pd.concat([df, new_row], ignore_index=True)
    # Try to read the existing data, or use the current df if file not found
    try:
        existing_df = pd.read_excel(file_path)
        # Combine existing data with new data
        updated_df = pd.concat([existing_df, df], ignore_index=True)
    except FileNotFoundError:
        # If the file does not exist, just use the new DataFrame
        updated_df = df

    # Write the DataFrame to an Excel file, overwriting the existing file
    updated_df.to_excel(file_path, index=False)

    return rotateAngle

# Joint Angle 1, Joint Angle 2 and Joint Angle 3 calculation
def landmarksJointAngle(handLandmarks):
    jointAngle = [mediumX, mediumY, mediumZ]
    wrist = handLandmarks.landmark[0]
    indexFingerMCP = handLandmarks.landmark[5]
    #data record
    data_filewrist.write(str(wrist) + '\n')
    data_filewrist.write('----------------\n')
    data_fileindex.write(str(indexFingerMCP) + '\n')
    data_fileindex.write('----------------\n')
    # calculate the distance between the wrist and the index finger
    palmSize = ((wrist.x - indexFingerMCP.x)**2 + (wrist.y - indexFingerMCP.y)**2 + (wrist.z - indexFingerMCP.z)**2)**0.5
    palmm = palmSize
    # calculate joint angle 1
    distance = palmSize
    angle = (wrist.x - indexFingerMCP.x) / distance  # calculate the radian between the wrist and the index finger
    angle = int(angle * 180 / 3.1415926)               # convert radian to degree
    angle1 = angle
    angle = limitRange(angle, anglePalmMin, anglePalmMedium)
    angle2 = angle
    jointAngle[0] = mapRange(angle, anglePalmMin, anglePalmMedium, maxX, minX)
    # calculate joint angle 2
    wrist_y = limitRange(wrist.y, minYWrist, maxYWrist)
    jointAngle[1] = mapRange(wrist_y, minYWrist, maxYWrist, maxY, minY)
    # calculate joint angle 3
    palmSize = limitRange(palmSize, minPalmSize, maxPalmSize)
    jointAngle[2] = mapRange(palmSize, minPalmSize, maxPalmSize, maxZ, minZ)
    # float to int (to prevent Dobot from keeping changeing its angle since it will increase the delay time)
    jointAngle = [int(i) for i in jointAngle]

    # Path to Excel file (data record)
    file_path = 'Record/right.xlsx'
    # Creating a single DataFrame with all required columns
    df = pd.DataFrame(columns=[
        'x(wrist)', 'y(wrist)', 'z(wrist)',
        'x(index)', 'y(index)', 'z(index)',
        'palm',
        'angle1', 'angle2',
        'joint1(sys)', 'joint2(sys)', 'joint3(sys)', 'palmm',
    ])
    # Create a new DataFrame for the current iteration's data
    new_row = pd.DataFrame({
        'x(wrist)': [wrist.x], 'y(wrist)': [wrist.y], 'z(wrist)': [wrist.z],
        'x(index)': [indexFingerMCP.x], 'y(index)': [indexFingerMCP.y], 'z(index)': [indexFingerMCP.z],
        'palm': [palmm],
        'angle1': [angle1], 'angle2': [angle2],
        'joint1(sys)': [jointAngle[0]], 'joint2(sys)': [jointAngle[1]], 'joint3(sys)': [jointAngle[2]],
        'palmm': [palmSize]
    })
    # Use concat to add the new row to the existing DataFrame
    df = pd.concat([df, new_row], ignore_index=True)
    # Try to read the existing data, or use the current df if file not found
    try:
        existing_df = pd.read_excel(file_path)
        # Combine existing data with new data
        updated_df = pd.concat([existing_df, df], ignore_index=True)
    except FileNotFoundError:
        # If the file does not exist, just use the new DataFrame
        updated_df = df

    # Write the DataFrame to an Excel file, overwriting the existing file
    updated_df.to_excel(file_path, index=False)


    return jointAngle

# Gripper Movement (the identification of first hand is needed, but the control approach is the same)
# Right hand to control the gripper to go down to grip, and left hand controls joint angle 4
def gripper(results):
    hands, imgage = detector.findHands(image)  # With Draw
    lmList, _ = detector.findPosition(image)  # Modify to receive lmList and _
    if len(hands) == 2:
        hand1 = hands[0]
        fingers1 = detector.fingersUp()
        handType1 = hand1["type"]  # Hand Type Left or Right
        if handType1 == "Right":
            hand2 = hands[1]
            handType2 = hand2["type"]  # Hand Type Left or Right
            #fingers2 = detector.fingersUpMulti(hand2)
            handLandmarks = results.multi_hand_landmarks[1] #Left hand (second detected)
            rotate = rotatAngle(handLandmarks)
            cv2.putText(image, str(rotate), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            # Measure the duration of executing the commands
            #start_time = time.time()
            dType.SetQueuedCmdStartExec(api)
            current_pose = dType.GetPose(api)
            dType.SetPTPCmdEx(api, 1, current_pose[0], current_pose[1], current_pose[2], rotate, 1)
            if fingers1 == [0,0,0,0,0]: # if right hand is closed, gripper goes down to grip
                dType.SetPTPCmdEx(api, 7, 0, 0, 10, 0, 1)
                dType.SetEndEffectorGripperEx(api, 1, 0)
                dType.SetPTPCmdEx(api, 7, 0, 0, (-30), 0, 1)
                dType.SetEndEffectorGripperEx(api, 1, 1)
                dType.SetPTPCmdEx(api, 7, 0, 0, 30, 0, 1)
                dType.SetEndEffectorGripperEx(api, 0, 0)
                dType.SetEndEffectorSuctionCupEx(api, 1, 1)
            pose1 = dType.GetPose(api)
            dType.SetQueuedCmdStopExec(api)
            # Stop the timer
            #end_time = time.time()

            # Path to Excel file
            file_path = 'Record/grip.xlsx'
            # Creating a single DataFrame with all required columns
            df = pd.DataFrame(columns=[
                'x(wrist)', 'y(wrist)', 'z(wrist)',
                'x(index)', 'y(index)', 'z(index)',
                'palm',
                'angle1', 'angle2', 'joint4(sys)'
            ])
            # Create a new DataFrame for the current iteration's data
            new_row = pd.DataFrame({
                'x': [pose1[0]], 'y(wrist)': [pose1[1]], 'z(wrist)': [pose1[2]],
                'Rotate': [pose1[3]], 'joint1': [pose1[4]], 'joint2': [pose1[5]],
                'joint3': [pose1[6]],
                'joint4': [pose1[7]], 'joint4(setting)': [rotate]
            })
            # Use concat to add the new row to the existing DataFrame
            df = pd.concat([df, new_row], ignore_index=True)
            # Try to read the existing data, or use the current df if file not found
            try:
                existing_df = pd.read_excel(file_path)
                # Combine existing data with new data
                updated_df = pd.concat([existing_df, df], ignore_index=True)
            except FileNotFoundError:
                # If the file does not exist, just use the new DataFrame
                updated_df = df

            # Write the DataFrame to an Excel file, overwriting the existing file
            updated_df.to_excel(file_path, index=False)

            # Calculate the elapsed time
            #elapsed_time = end_time - start_time
            #print(" ", elapsed_time)
            #data2.write(str(elapsed_time) + "\n")

            #print("Left: ", fingers2)

        elif handType1 == "Left":
            hand2 = hands[1]
            #handType2 = hand2["type"]  # Hand Type Left or Right
            fingers2 = detector.fingersUpMulti(hand2)
            handLandmarks = results.multi_hand_landmarks[0] # Left hand (First detected)
            rotate = rotatAngle(handLandmarks)
            cv2.putText(image, str(rotate), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            #print("Left1: ", fingers1)
            #print("Right1: ", fingers2)
            # Measure the duration of executing the commands
            start_time = time.time()
            dType.SetQueuedCmdStartExec(api)
            current_pose = dType.GetPose(api)
            dType.SetPTPCmdEx(api, 1, current_pose[0], current_pose[1], current_pose[2], rotate, 1)
            if fingers2 == [0, 0, 0, 0, 0]: # if right hand is closed, gripper goes down to grip
                dType.SetEndEffectorGripperEx(api, 1, 0)
                dType.SetPTPCmdEx(api, 7, 0, 0, (-30), 0, 1)
                dType.SetEndEffectorGripperEx(api, 1, 1)
                dType.SetPTPCmdEx(api, 7, 0, 0, 30, 0, 1)
                dType.SetEndEffectorGripperEx(api, 0, 0)
            dType.SetQueuedCmdStopExec(api)
            # Stop the timer
            end_time = time.time()

            # Calculate the elapsed time
            elapsed_time = end_time - start_time
            print(" ", elapsed_time)
            data1.write(str(elapsed_time) + "\n")


    cv2.imshow('MediaPipe Hands', image)


# Suction Cup Movement (the identification of first hand is needed, but the control approach is the same)
# Right hand to control the gripper to go down to grip, and left hand controls joint angle 4
def suction(results):
    hands, imgage = detector.findHands(image)  # With Draw
    lmList, _ = detector.findPosition(image)  # Modify to receive lmList and _
    if len(hands) == 2:
        hand1 = hands[0]
        fingers1 = detector.fingersUp()
        handType1 = hand1["type"]  # Hand Type Left or Right
        if handType1 == "Right":
            hand2 = hands[1]
            #handType2 = hand2["type"]  # Hand Type Left or Right
            #fingers2 = detector.fingersUpMulti(hand2)

            print("Right: ", fingers1)
            handLandmarks = results.multi_hand_landmarks[1] #Left hand (second detected)
            # print("RightFirst", handLandmarks)
            # print("RightFirstRe", results.multi_hand_landmarks)
            rotate = rotatAngle(handLandmarks)
            cv2.putText(image, str(rotate), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            dType.SetQueuedCmdStartExec(api)
            current_pose = dType.GetPose(api)
            dType.SetPTPCmdEx(api, 1, current_pose[0], current_pose[1], current_pose[2], rotate, 1)
            if fingers1 == [0,0,0,0,0]: # if right hand is closed, gripper goes down to grip
                dType.SetEndEffectorGripperEx(api, 1, 0)
                dType.SetPTPCmdEx(api, 7, 0, 0, (-15), 0, 1)
                dType.SetEndEffectorSuctionCupEx(api, 1, 1)
                dType.SetPTPCmdEx(api, 7, 0, 0, 15, 0, 1)

        elif handType1 == "Left":
            hand2 = hands[1]
            #handType2 = hand2["type"]  # Hand Type Left or Right
            fingers2 = detector.fingersUpMulti(hand2)
            handLandmarks = results.multi_hand_landmarks[0] # Left hand (First detected)
            # print("LeftFirst", handLandmarks)
            # print("LeftFirstRe", results.multi_hand_landmarks)
            rotate = rotatAngle(handLandmarks)
            cv2.putText(image, str(rotate), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            #print("Left1: ", fingers1)
            print("Right1: ", fingers2)
            dType.SetQueuedCmdStartExec(api)
            current_pose = dType.GetPose(api)
            dType.SetPTPCmdEx(api, 1, current_pose[0], current_pose[1], current_pose[2], rotate, 1)
            if fingers2 == [0, 0, 0, 0, 0]: # if right hand is closed, gripper goes down to grip
                dType.SetEndEffectorGripperEx(api, 1, 0)
                dType.SetPTPCmdEx(api, 7, 0, 0, (-15), 0, 1)
                dType.SetEndEffectorSuctionCupEx(api, 1, 1)
                dType.SetPTPCmdEx(api, 7, 0, 0, 15, 0, 1)

    cv2.imshow('MediaPipe Hands', image)


with mpHands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)


        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        #Joint angles calculation
        if results.multi_hand_landmarks:

            if len(results.multi_hand_landmarks) == 1:
                handLandmarks = results.multi_hand_landmarks[0]
                jointAngle = landmarksJointAngle(handLandmarks)
                # if the angles are the same as previous angles, Dobot would not move (Decrease the delay time)
                if jointAngle != jointAnglePrev:
                    jointAnglePrev = jointAngle
            else:

                gripper(results) # Gripper Movement
                # suction(results) # Suction Cup Movement
            # Mediapipe Hand Landmarks
            for hand_landmarks in results.multi_hand_landmarks:
                mpDrawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mpHands.HAND_CONNECTIONS,
                    mpDrawingStyles.get_default_hand_landmarks_style(),
                    mpDrawingStyles.get_default_hand_connections_style())

        # show joint angle
        cv2.putText(image, str(jointAngle), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        # Calculate FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(image, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        data.write(str(fps) + '\n')
        cv2.imshow('MediaPipe Hands', image)
        # Measure the duration of executing the commands
        #start_time = time.time()
        dType.SetQueuedCmdStartExec(api)
        current_pose = dType.GetPose(api)
        dType.SetPTPCmdEx(api, 4, jointAngle[0], jointAngle[2], jointAngle[1], current_pose[7], 1)
        pose = dType.GetPose(api)
        dType.SetQueuedCmdStopExec(api)
        # Stop the timer
        end_time = time.time()

        if write_video:
            out.write(image)
        if cv2.waitKey(5) & 0xFF == 27:

            if write_video:
                 out.release()
                 break
cap.release()
dType.SetEndEffectorGripperEx(api, 0, 0)
dType.SetQueuedCmdStopExec(api)