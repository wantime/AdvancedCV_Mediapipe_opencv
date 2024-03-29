import cv2 as cv
import mediapipe as mp
import time


class faceDetector():
    def __init__(self, minDetectionConf=0.5, model=1):
        self.minDetectionConf = minDetectionConf
        self.model = model

        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionConf, self.model)

    def findFaces(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)

        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                bboxs.append([id, bbox, detection.score])

                if draw:
                    img = self.fancyDraw(img, bbox,)
                    #cv.rectangle(img, bbox, (255, 0, 255), 2)
                    cv.putText(img, f'{int(detection.score[0] * 100)}%',
                               (bbox[0], bbox[1] - 20), cv.FONT_HERSHEY_PLAIN,
                               2, (255, 0, 255), 2)

        return img, bboxs

    def findPosition(self, img, draw=True):

        bboxList = []

        if self.results:
            for id, detection in enumerate(self.results.detection):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                bboxList.append(bbox)
                if draw:
                    cv.rectangle(img, bbox, (0, 255, 0), 2)
        return bboxList

    def fancyDraw(self, img, bbox, l=30, t=5):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h
        color = (255, 0, 255)
        cv.rectangle(img, bbox, color, 2)
        # Top Left x,y
        cv.line(img, (x,y), (x+l, y), color, t)
        cv.line(img, (x,y), (x, y+l), color, t)
        # Top Right x1,y
        cv.line(img, (x1, y), (x1 - l, y), color, t)
        cv.line(img, (x1, y), (x1, y + l), color, t)
        # Bottom Right x1,y1
        cv.line(img, (x1, y1), (x1 - l, y1), color, t)
        cv.line(img, (x1, y1), (x1, y1 - l), color, t)
        # Bottom left x,y1
        cv.line(img, (x, y1), (x + l, y1), color, t)
        cv.line(img, (x, y1), (x, y1 - l), color, t)
        return img

def main():
    cap = cv.VideoCapture('video.mp4')
    pTime = 0
    detector = faceDetector(0.4)
    while True:
        success, img = cap.read()
        img, bboxs = detector.findFaces(img)
        print(bboxs)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv.putText(img, f'FPS:{int(fps)}', (20, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)
        cv.imshow("face detection", img)
        cv.waitKey(1)


if __name__ == "__main__":
    main()
