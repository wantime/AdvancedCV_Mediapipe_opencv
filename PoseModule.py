import cv2 as cv
import mediapipe as mp
import time
import math

class poseDetector:

    def __init__(self,
                 static_image_mode=False,
                 model_complexity=1,
                 smooth_landmarks=True,
                 enable_segmentation=False,
                 smooth_segmentation=True,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=static_image_mode,
                                     model_complexity=model_complexity,
                                     smooth_landmarks=smooth_landmarks,
                                     enable_segmentation=enable_segmentation,
                                     smooth_segmentation=smooth_segmentation,
                                     min_detection_confidence=min_detection_confidence,
                                     min_tracking_confidence=min_tracking_confidence)

    def find_pose(self, img, draw=True):

        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def find_position(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv.circle(img, (cx, cy), 5, (255, 0, 0), cv.FILLED)
        return self.lmList, img

    def find_angle(self, img, p1, p2, p3, draw=True):

        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        # calculate the angle
        angle = math.degrees(math.atan2(y3-y2, x3-x2) - math.atan2(y1-y2, x1-x2))

        while angle < 0:
            angle = angle + 180
        #print(angle)
        if draw:
            cv.line(img, (x1,y1), (x2, y2), (255, 255, 255), 3)
            cv.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv.circle(img, (x1, y1), 10, (0, 0, 255), cv.FILLED)
            cv.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv.circle(img, (x2, y2), 10, (0, 0, 255), cv.FILLED)
            cv.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv.circle(img, (x3, y3), 10, (0, 0, 255), cv.FILLED)
            cv.circle(img, (x3, y3), 15, (0, 0, 255), 2)
            cv.putText(img, str(int(angle)), (x2-20, y2+50),
                       cv.FONT_HERSHEY_PLAIN, 2, (255, 155, 0), 2)
        return angle


def main():
    cap = cv.VideoCapture()
    pTime = 0
    detector = poseDetector()
    while True:
        success, img = cap.read()
        img = cv.imread('data/pose/1.jpg')
        img = detector.find_pose(img)
        lmList, img = detector.find_position(img)
        if len(lmList) != 0:
            print(lmList[14])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv.putText(img, str(int(fps)), (70, 50), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv.imshow("Image", img)
        cv.waitKey(1)
    # imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)


if __name__ == "__main__":
    main()
    # detector = mp.solutions.pose
    # x = detector.Pose()
    # img = cv.imread('data/pose/1.jpg')
    # imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # mpDraw = mp.solutions.drawing_utils
    # results = x.process(imgRGB)
    # if results.pose_landmarks:
    #    mpDraw.draw_landmarks(img, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
    #
    # cv.imshow('img', img)
    # cv.waitKey(0)
