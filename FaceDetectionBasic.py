import cv2 as cv
import mediapipe as mp
import time

cap = cv.VideoCapture("")
pTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection()

while True:
    success, img = cap.read()

    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)

    if results.dectections:
        for id, detection in enumerate(results.detections):
            print(id, detection)
            mpDraw.draw_detection(img)

    cTime = time.time()
    fps = 1 / (cTime-pTime)
    pTime = cTime
    cv.putText(img, f'FPS:{int(fps)}', (20,70), cv.FONT_HERSHEY_PLAIN, 3, (255,0,255), 2)
    faceDetection.process(img)


    cv.imshow("face", img)
    cv.waitKey(10)