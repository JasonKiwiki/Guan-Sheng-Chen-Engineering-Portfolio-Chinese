################################################################################
# Reference: https://www.dobot-robots.com/service/download-center
# Author: Guan-Sheng Chen
# Date: 9 May 2024
# Copyright, Guan-Sheng Chen 2024
################################################################################
# The file is to control Dobot Magician to draw a circle, line and symbol "X"
# The task of gesture "Draw"
################################################################################
import threading
import DobotDllType as dType
import math
from numbers import Number
import sys
#Dobot connection status identification from Dobot Demo package
CON_STR = {
    dType.DobotConnect.DobotConnect_NoError: "DobotConnect_NoError",
    dType.DobotConnect.DobotConnect_NotFound: "DobotConnect_NotFound",
    dType.DobotConnect.DobotConnect_Occupied: "DobotConnect_Occupied"}

# Load Dll and get the CDLL object
api = dType.load()
# Set ARC parameters
xyzVelocity = 100
rVelocity = 100
xyzAcceleration = 100
rAcceleration = 100

#Length of pattern transction from other file (sys library imported necessary)
def valueObtain():
    # Retrieve the values of tran from the command-line arguments
    if len(sys.argv) > 2:
        instruct = int(sys.argv[1])
        value = float(sys.argv[2])
        print(f"The values are: {instruct}, {value}")
    else:
        print("Not enough command-line arguments provided.")
        instruct = 0
        value = 0
    return instruct, value
# Symbol "X" Drawing
def movementX(api, length):
    #Symbol "X" point calculation
    sqarelength = (math.sqrt(2) * length) / 2
    homeX = dType.GetPoseEx(api, 1)
    homeY = dType.GetPoseEx(api, 2)
    homeZ = dType.GetPoseEx(api, 3)
    dType.SetPTPCmdEx(api, 0, homeX, homeY, homeZ, 0, 1) #start the movement of Dobot
    x1 = homeX + (length / 2) * math.sin(45 / 180.0 * math.pi)
    y1 = homeY + (length / 2) * math.cos(45 / 180.0 * math.pi)
    current_pose = dType.GetPose(api)#identify the current position (x,y,z,R,Joint1,Joint2,Joint3,Joint4)
    dType.SetPTPCmdEx(api, 0, x1, y1, homeZ, current_pose[3], 1)#PTP movement
    current_pose = dType.GetPose(api)
    dType.SetPTPCmdEx(api, 1, (x1 - sqarelength), (y1 - sqarelength), homeZ, current_pose[3], 1)
    dType.SetPTPCmdEx(api, 0, x1, (y1 - sqarelength), homeZ, 0, 1)
    current_pose = dType.GetPose(api)
    dType.SetPTPCmdEx(api, 1, (x1 - sqarelength), y1, homeZ, current_pose[3], 1)

# Circle Drawing (Center point of circle is the current position of Dobot)
def circleMovement(api, iterations, radius):
    current_pose = dType.GetPose(api)
    #first point of the circle calculation
    angle = 0
    X1 = current_pose[0] + radius * math.cos(angle / 180.0 * math.pi)
    Y1 = current_pose[1] + radius * math.sin(angle / 180.0 * math.pi)
    while angle != 360:
        X = current_pose[0] + radius * math.cos(angle / 180.0 * math.pi)
        Y = current_pose[1] + radius * math.sin(angle / 180.0 * math.pi)
        #current_pose = dType.GetPose(api)
        dType.SetPTPCmdEx(api, 1, X, Y, current_pose[2], current_pose[3], 1)
        #dType.dSleep(0)
        angle = (angle if isinstance(angle, Number) else 0) + iterations
    current_pose = dType.GetPose(api)
    dType.SetPTPCmdEx(api, 1, X1, Y1, current_pose[2], current_pose[3], 1)
    current_pose = dType.GetPose(api)
    dType.SetPTPCmdEx(api, 1, 260, 35, 50, current_pose[3], 1)

# Line drawing
def lineMovement(api, length):
    # Get current pose
    current_pose = dType.GetPose(api)#starting point is the current Dobot position
    dType.SetPTPCmdEx(api, 1, current_pose[0], current_pose[1]+ length, current_pose[2], current_pose[3], 1)
# Connect Dobot
state = dType.ConnectDobot(api, "", 115200)[0]
print("Connect status:", CON_STR[state])


#if Dobot connected
if (state == dType.DobotConnect.DobotConnect_NoError):

    # Clean Command Queued
    dType.SetQueuedCmdClear(api)#clear the previous command to execute new command

    # Async Motion Params Setting
    dType.SetHOMEParams(api, 200, 200, 200, 200, isQueued=1)
    dType.SetPTPJointParams(api, 200, 200, 200, 200, 200, 200, 200, 200, isQueued=1)
    dType.SetPTPCommonParams(api, 100, 100, isQueued=1)

    # Async Home
    #dType.SetHOMECmd(api, temp=0, isQueued=1) #Dobot calibration
    # Start to Execute Command Queue
    dType.SetQueuedCmdStartExec(api)#without this code, the Dobot would be unable to control
    instruct, value = valueObtain()
    value = value / 10 # to prevent the length of pattern is too large to cause error for Dobot
    if instruct == 2:
        circleMovement(api, 1, value)
        print("Circle")
    elif instruct == 3:
        movementX(api, value)
        print("X Movement")
    elif instruct == 1:
        lineMovement(api, value)
        print("Line")
    else:
        print("Fail")

    dType.SetQueuedCmdStopExec(api)


# Disconnect Dobot
dType.DisconnectDobot(api)
