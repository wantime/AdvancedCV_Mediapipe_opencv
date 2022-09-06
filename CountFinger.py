import os
import time
import HandsTrackingModule as htm
import cv2 as cv

# prepare the image of numbers
path = './data/finger'
imgList = os.listdir(path)
fingerList = []
for img_file in imgList:
    img_path = path + '/' + img_file
    fingerList.append(img_path)

# prepare the web-camera
wCam, hCam = 640, 480
cap = cv.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

# prepare the hand detector
handDetector = htm.handDetector()
handsPointIndex = [4, 8, 12, 16, 20]

# continuous acquire image
while True:
    # success, img = cap.read()
    img = cv.imread('data/hand.jpg')
    img = handDetector.findHands(img)
    lmList = handDetector.findPosition(img)
    results = []
    if len(lmList) > 0:
        if lmList[handsPointIndex[0]][1] < lmList[handsPointIndex[0]-2][1]:
            results.append(1)
        else:
            results.append(0)
        for id in range(1,5):
            if lmList[handsPointIndex[id]][2] < lmList[handsPointIndex[id]-2][2]:
                results.append(1)
            else:
                results.append(0)

        number = results.count(1)
        print(number)
        print(fingerList[number])
        img_finger = cv.imread(fingerList[number])

        iw, ih, ic = img_finger.shape
        img[:iw, :ih, :] = img_finger

        cv.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv.FILLED)
        cv.putText(img, str(number), (45, 375), cv.FONT_HERSHEY_PLAIN,
                   10, (255, 0, 0), 25)


    cTime = time.time()
    fps = 1 / (cTime-pTime)
    pTime = cTime


    cv.putText(img, f'FPS:{int(fps)}', (440, 50), cv.FONT_HERSHEY_PLAIN,
               3, (0, 255, 0), 2)
    cv.imshow('image', img)

    cv.waitKey(1)

