import cv2 as cv
import mediapipe as mp
import time


class faceDetection():
    def __init__(self, minDetectionConf=0.5, model=1):
        self.minDetectionConf = minDetectionConf
        self.model = model

        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionConf, self.model)

    def findFace(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)

        if self.results.detections:
            if draw:
                for id, detection in enumerate(results.detection):
                    self.mpDraw.draw_detection(img, detection)
        return img

    def findPosition(self, img, draw=True):

        bboxList = []

        if self.results:
            for id, detection in enumerate(results.detection):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                bboxList.append(bbox)
                if draw:
                    cv.rectangle(img, bbox, (0, 255, 0), 2)
        return bboxList


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
            cv.putText(img, f'{int(detection.score[0])}', (bbox[0], bbox[1] - 20),
                       cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)


def main():
    cap = cv.VideoCapture()
    pTime = 0

    while True:
        success, img = cap.read()

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv.putText(img, f'FPS:{int(fps)}', (20, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)
        cv.imshow("face detection", img)
        cv.waitKey(1)


if __name__ == "__main__":
    main()
