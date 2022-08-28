import time


import cv2 as cv
import PoseModule as pm


cap = cv.VideoCapture()
wCam, hCam = 640, 480
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

# initial a detector
poseDetector = pm.poseDetector()

while True:
    success, img = cap.read()
    img = cv.imread('data/pose/2.jpg')
    # try to detect
    img = poseDetector.find_pose(img, draw=False)
    # get the land mark (the coordinates of pose)
    lmList, img = poseDetector.find_position(img, draw=False)
    # do something through the relative position of points.
    if len(lmList) > 0:
        #right arm
        poseDetector.find_angle(img, 12, 14, 16)
        #left arm
        poseDetector.find_angle(img, 11, 13, 15)


    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv.putText(img, f'FPS:{int(fps)}', (40, 50), cv.FONT_HERSHEY_PLAIN,
               3, (0, 255, 0), 2)
    cv.imshow('camera', img)
    cv.waitKey(0)