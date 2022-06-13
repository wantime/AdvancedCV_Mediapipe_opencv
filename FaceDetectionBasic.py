import cv2 as cv
import mediapipe as mp
import time

cap = cv.VideoCapture("video.mp4")
pTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(0.75)

while True:
    success, img = cap.read()

    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)

    if results.detections:
        for id, detection in enumerate(results.detections):
            # print(id, detection)
            # print(detection.score)
            # print(detection.location_data.relative_bouding_box)
            # mpDraw.draw_detection(img, detection)
            print(detection.location_data.relative_bounding_box.xmin)
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                   int(bboxC.width * iw), int(bboxC.height * ih)

            cv.rectangle(img, bbox, (0, 255, 0), 2)

    cTime = time.time()
    fps = 1 / (cTime-pTime)
    pTime = cTime
    cv.putText(img, f'FPS:{int(fps)}', (20,70), cv.FONT_HERSHEY_PLAIN, 3, (255,0,255), 2)
    faceDetection.process(img)


    cv.imshow("face", img)
    cv.waitKey(1)