import cv2
import mediapipe as mp
draw=mp.solutions.drawing_utils
holistic=mp.solutions.holistic
cap=cv2.VideoCapture(0)
with holistic.Holistic() as hol:
    while True:
        s,cam=cap.read()
        image=cv2.cvtColor(cam,cv2.COLOR_BGR2RGB)
        results=hol.process(image)
        draw.draw_landmarks(
            cam,results.face_landmarks,holistic.FACEMESH_CONTOURS)
        cv2.imshow("FACE",cam)
        cv2.waitKey(1)
cap.release()


