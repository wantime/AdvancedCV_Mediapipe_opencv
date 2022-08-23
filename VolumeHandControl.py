import math

import cv2 as cv
import time
import numpy as np
import HandsTrackingModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
################################
wCam, hCam = 640, 480
################################
cap = cv.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

detector = htm.handDetector(minDectionConfidence=0.7)


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar = 400
volPer = 0
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmLise = detector.findPosition(img, draw=False)
    if len(lmLise) != 0:
        # print(lmLise[4], lmLise[8])

        x1, y1 = lmLise[4][1], lmLise[4][2]
        x2, y2 = lmLise[8][1], lmLise[8][2]
        cx, cy = (x1+x2) // 2, (y1 + y2) // 2


        cv.circle(img, (x1, y1), 10, (0, 255, 0), cv.FILLED)
        cv.circle(img, (x2, y2), 10, (0, 255, 0), cv.FILLED)
        cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv.circle(img, (cx, cy), 10, (0, 255, 0), cv.FILLED)

        distance = math.hypot(x2-x1, y2-y1)
        print(distance)

        # Hand Range 50 - 260
        # Volume Range -65 - 0
        vol = np.interp(distance, [50, 260], [minVol, maxVol])
        volBar = np.interp(distance, [50, 260], [400, 150])
        volPer = np.interp(distance, [50, 300], [0, 100])
        volume.SetMasterVolumeLevel(vol, None)
        if distance < 50:
            cv.circle(img, (cx, cy), 10, (255, 0, 255), cv.FILLED)

    cv.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    cv.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv.FILLED)
    cv.putText(img, f'{int(volPer)} %', (40, 450), cv.FONT_HERSHEY_COMPLEX,
               1, (0, 0, 250), 3)
    # calculate the distance between No.4 and No.8



    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv.putText(img, f'FPS:{int(fps)}', (40,50), cv.FONT_HERSHEY_PLAIN,
               3, (0,255,0), 2)

    cv.imshow('Img', img)
    cv.waitKey(1)