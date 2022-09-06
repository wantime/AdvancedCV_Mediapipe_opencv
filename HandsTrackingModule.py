import cv2 as cv
import mediapipe as mp
import time


class handDetector:
    def __init__(self,
                 mode=False,
                 maxNum=2,
                 complexity=1,
                 minDectionConfidence=0.5,
                 minTrackingConfidence=0.5):
        self.mode = mode
        self.maxNum = maxNum
        self.complexity = complexity
        self.minDectionConfidence = minDectionConfidence
        self.minTrackingConfidence = minTrackingConfidence

        self.mphands = mp.solutions.hands
        self.hands = self.mphands.Hands(self.mode,
                                        self.maxNum,
                                        self.complexity,
                                        self.minDectionConfidence,
                                        self.minTrackingConfidence)
        self.mpDraw = mp.solutions.drawing_utils
        self.mpDraw.DrawingSpec([0, 255, 0], thickness=1, circle_radius=2)
    def findHands(self,
                  img,
                  draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mphands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):

        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv.circle(img, (cx, cy), 5, (0, 255, 0), cv.FILLED)

        return self.lmList

    def fingerUp(self):
        handsPointIndex = [8, 12, 16, 20]
        results = []

        if len(self.lmList) > 0:
            if self.lmList[handsPointIndex[0]][1] < self.lmList[handsPointIndex[0] - 2][1]:
                results.append(1)
            else:
                results.append(0)
            for id in range(1, 5):
                if self.lmList[handsPointIndex[id]][2] < self.lmList[handsPointIndex[id] - 2][2]:
                    results.append(1)
                else:
                    results.append(0)
        return results


def main():
    pTime = 0
    cTime = 0
    cap = cv.VideoCapture(0)
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findHands(img)
        if len(lmList) != 0:
            print(lmList[4])
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv.putText(img, str(int(fps)), (10, 78), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv.imshow(img)
        cv.waitKey(1)


if __name__ == "__main__":
    main()
