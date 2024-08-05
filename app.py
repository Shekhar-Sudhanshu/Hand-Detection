import cv2
import mediapipe as mp
import time

capture = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

prevTime = 0

while True:
    success, img = capture.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lms in enumerate(handLms.landmark):

                h, w, c = img.shape
                cx, cy = int(lms.x * w), int(lms.y * h)
                #print(id, cx, cy)

                if(id == 0):
                    cv2.circle(img, (cx, cy), 15, (255, 255, 255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    currTime = time.time()
    fps = 1/(currTime - prevTime)
    prevTime = currTime

    cv2.putText(img, f"FPS - {int(fps)}", (20, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 255, 255), 2)
    cv2.putText(img, "Press 'q' to exit", (28, 62), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1)

    cv2.imshow("Img", img)

    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break