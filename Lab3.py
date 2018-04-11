import cv2
import numpy as np
import math


threshold = (0.1, 1)

imgGRY = cv2.cvtColor(cv2.imread("images/test.jpg"), cv2.COLOR_BGR2GRAY)
img_result_GRY = np.copy(imgGRY)
histogram = np.zeros((255, 255), np.uint8)

hist = cv2.calcHist([imgGRY], [0], None, [256], [0, 256])
print(hist)

for i in range(img_result_GRY.shape[0]):
    for j in range(img_result_GRY.shape[1]):
        if not (img_result_GRY[i, j] < threshold[0] * 255 or img_result_GRY[i, j] > threshold[1] * 255):
            img_result_GRY[i, j] -= threshold[0]
            img_result_GRY[i, j] *= 1 / (threshold[1] - threshold[0])

cv2.imshow("Histogram", histogram)
cv2.imshow("Original", imgGRY)
cv2.imshow("Result", img_result_GRY)

cv2.waitKey(0)
cv2.destroyAllWindows()