import cv2
import numpy as np
import math

#   Img Width
w = 512

#   Create Canvas
imgHSV = np.zeros((w, w, 3), np.uint8)

#   Draw
for i in range(w):
    for j in range(w):
        h = 255 * (math.fabs(math.sin((i + j) * 0.5)) * 0.5 + 0.5) * \
            ((math.fabs(math.sin(math.radians(i * 5 * (w / 360)))) ** 2 / 2) +
             (math.fabs(math.cos(math.radians(90 + j * 5 * (w / 360)))) ** 2 / 2))
        s = 255
        v = 255
        imgHSV[i, j] = (h, s, v)

#   Convert to BGR
imgBRG = cv2.cvtColor(imgHSV, cv2.COLOR_HSV2BGR)

#   Dran'n'Save
cv2.imwrite('renders/Lab1.png', imgBRG)
cv2.imshow('Lab one - BRG', imgBRG)

cv2.waitKey(0)
cv2.destroyAllWindows()
