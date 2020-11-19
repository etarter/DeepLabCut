import numpy as np
import cv2

cap = cv2.VideoCapture(r'C:\Users\etarter\Downloads\videos-multianimal\1_2020-06-14_21-02-42.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame',gray)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
