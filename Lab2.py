import cv2
import numpy as np
import math

threshold = 255 / 2

imgRGB = cv2.imread("images/test.jpg")
imgHSV = cv2.cvtColor(imgRGB, cv2.COLOR_BGR2HSV)
imgBIN = np.zeros((imgHSV.shape[0], imgHSV.shape[1], 1), np.uint8)

for i in range(imgHSV.shape[0]):
    for j in range(imgHSV.shape[1]):
        if imgHSV[i, j, 2] > threshold:
            imgBIN[i, j, 0] = 255


cv2.imwrite('renders/Lab2.png', imgBIN)
cv2.imshow("Original", imgRGB)
cv2.imshow("Result", imgBIN)

cv2.waitKey(0)
cv2.destroyAllWindows()
