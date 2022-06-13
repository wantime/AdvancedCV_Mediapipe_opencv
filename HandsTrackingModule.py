import cv2 as cv
import mediapipe as mp
import time

class handDetector():
    def __init__(self,
                 mode,
                 maxNum,
                 complexity,
                 minDectionConfidence,
                 minTrackingConfidence):
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

    def findHands(self,
                  img,
                  draw=True):
        imgBGR = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        self.results = self.hands.process(imgBGR)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mphands.HAND_CONNECTIONS)
        return img


        def findPosition(img, handNo=0, draw=True):

            lmList = []
            if self.results.multi_hand_landmarks:
                myHand = self.results.multi_hand_landmarks[handNo]
                for id, lm in enumerate(myHand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x*w), int(lm.y*h)
                    lmList.append([id, cx, cy])
                    if draw:
                        cv.circle(img, (cx, cy), 15, (255, 0, 255), cv.FILLED)

            return lmList

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