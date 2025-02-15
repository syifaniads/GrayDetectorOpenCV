import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # width = int(cap.get(3))
    # height = int(cap.get(4))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_gray = np.array([0, 0, 50])
    upper_gray= np.array([180,50,200])

    mask = cv2.inRange(hsv, lower_gray, upper_gray)
    result = cv2.bitwise_and(frame,frame, mask=mask)

    # 1 1 = 1
    # 0 1 = 0
    # 1 0 = 0
    # 0 0 = 0

    cv2.imshow('Original', frame)
    cv2.imshow('Gray Mask', mask)
    cv2.imshow('Gray Detection', result)

    # cv2.imshow('frame', result)
    # cv2.imshow('mask', mask)

    if cv2.waitKey(1)==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



# BGR_color= np.array([[[255,0,0,0]]])
# x = cv2.cvtColor([[[255,0,0]]], cv2.COLOR_BGR2HSV)
# x[0][0]