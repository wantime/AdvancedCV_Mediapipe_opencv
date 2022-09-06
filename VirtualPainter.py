import numpy as np

import HandsTrackingModule as htm
import cv2 as cv
import time

cap = cv.VideoCapture(0)

wCam, hCam = 640, 480
cap = cv.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

handDetector = htm.handDetector()

ColorsBar = [(100, 0, 0), (100, 200, 0), (0, 100, 0), (0, 200, 100), (0, 0, 100)]
ColorSwitch = 0
paintColor = ColorsBar[ColorSwitch]
px, py = 0, 0

canvas = np.zeros(shape=(wCam, hCam, 3), dtype=np.uint8)
while True:
    success, img = cap.read()
    img = cv.flip(img, 1)
    img, lmList = handDetector.findPosition(img)
    fingerUp = handDetector.fingerUp()
    numOfFingerUp = fingerUp.sum()

    if numOfFingerUp == 1 and fingerUp[0] == 1:
        # painting
        x, y = lmList[8][1], lmList[8][2]
        if px == 0 and py == 0:
            px, py = x, y
        cv.line(canvas, (px, py), (x, y), paintColor, 1)
    else:
        px, py = 0, 0
    if numOfFingerUp == 0:
        # change color
        ColorSwitch = ColorSwitch + 1
        if ColorSwitch == len(ColorsBar):
            ColorSwitch = 0
        paintColor = ColorsBar[ColorSwitch]
    elif numOfFingerUp >= 4:
        # eraser
        pass
        ex1, ey1 = lmList[8][1], lmList[8][2]
        ex2, ey2 = lmList[17][1], lmList[17][2]
        cv.rectangle(canvas, (ex1, ey1), (ex2, ey2), (0, 0, 0))

    # canvas
    canvasGray = cv.cvtColor(canvas, cv.COLOR_BGR2GRAY)
    __, canInv = cv.threshold(canvasGray, 50, 255, cv.THRESH_BINARY_INV)
    imgInv = cv.cvtColor(canInv, cv.COLOR_GRAY2BGR)
    img = cv.bitwise_and(img,imgInv)
    img = cv.bitwise_or(img, canInv)


    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv.putText(img, f'FPS:{int(fps)}', (440, 50), cv.FONT_HERSHEY_PLAIN,
               3, (0, 255, 0), 2)
    cv.imshow('painting', img)

    cv.waitKey(0)