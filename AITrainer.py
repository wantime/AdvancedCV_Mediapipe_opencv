import time

import cv2 as cv
import numpy as np

import PoseModule as pm

cap = cv.VideoCapture()
wCam, hCam = 640, 480
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

# initial a detector
poseDetector = pm.poseDetector()
count = 0
# arm direction: 1->up 0->down
direction = 1
barColor = (0, 255, 0)
while True:
    success, img = cap.read()
    img = cv.imread('data/pose/2.jpg')
    img = cv.resize(img, dsize=(650, 490))
    # try to detect
    img = poseDetector.find_pose(img, draw=False)
    # get the land mark (the coordinates of pose)
    lmList, img = poseDetector.find_position(img, draw=False)
    # do something through the relative position of points.
    if len(lmList) > 0:
        # right arm
        # poseDetector.find_angle(img, 12, 14, 16)
        # left arm
        angle = poseDetector.find_angle(img, 11, 13, 15)
        print(angle)
        precent = int(np.interp(angle, [20, 180], [0, 100]))
        print(precent)
        bar = int(np.interp(angle, [20, 180], [440, 80]))
        print(bar)
        if precent == 100:
            if dir == 1:
                count += 0.5
                dir = 0
                barColor = (0, 0, 200)
        if precent == 0:
            if dir == 0:
                count += 0.5
                dir = 1
                barColor = (0, 255, 0)
        # bar
        cv.putText(img, f'{str(precent)}%', (580, 70), cv.FONT_HERSHEY_PLAIN,
                   2, barColor, 2)
        cv.rectangle(img, (580, 80), (640, 440), barColor, 2)
        cv.rectangle(img, (580, bar), (640, 440), barColor, cv.FILLED)
    # count
    #cv.rectangle(img, (0, 380), (100, 480), (0, 255, 0), cv.FILLED)
    #cv.putText(img, f'{str(int(count))}', (40, 430), cv.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 5)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv.putText(img, f'FPS:{int(fps)}', (40, 50), cv.FONT_HERSHEY_PLAIN,
               3, (0, 255, 0), 2)
    cv.imshow('camera', img)
    cv.waitKey(0)
