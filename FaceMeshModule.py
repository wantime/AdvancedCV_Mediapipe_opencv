import cv2 as cv
import mediapipe as mp
import time

class FaceMeshDetector():

    def __init__(self, staticMode=False,
                 maxFaces=2,
                 minDetectionCon=0.5,
                 minTrackCon=0.5):
        self.static_image_mode = staticMode
        self.max_faces = maxFaces
        self.min_detection_con = minDetectionCon
        self.min_track_con = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(max_num_faces=3)
        self.drawSpec = self.mpDraw.DrawingSpec([0,255,0], thickness=1, circle_radius=2)

    def findFaceMesh(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS,
                                      self.drawSpec, self.drawSpec)
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    # if id < 10:
                    #     cv.putText(img, str(id), (x, y), cv.FONT_HERSHEY_PLAIN,
                    #            1, (0, 255, 0), 1)
                    face.append([x, y])
                faces.append(face)
        return img, faces


def main():
    cap = cv.VideoCapture('video.mp4')
    pTime = 0
    detector = FaceMeshDetector()
    while True:
        success, img = cap.read()
        img, faces = detector.findFaceMesh(img)
        if len(faces)>0:
            print(faces[0])
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv.putText(img, f'FPS:{int(fps/10)*10}', (20,70), cv.FONT_HERSHEY_PLAIN,
                   3, (0, 255, 0), 3)
        cv.imshow('Image', img)
        cv.waitKey(1)

if __name__ == '__main__':
    main()