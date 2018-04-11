import cv2
import numpy as np
import math

imgRGB = cv2.imread("images/test2.jpg")
imgGRY = cv2.cvtColor(imgRGB, cv2.COLOR_BGR2GRAY)

kernel = np.array([-0.5, 0.5])

kernel_box = 1 / 9 * np.array([[1, 1, 1],
                                [1, 1, 1],
                                [1, 1, 1]])

kernel_another = -1 / 9 * np.array([[1, 1, 1],
                                [1, -8, 1],
                                [1, 1, 1]])

kernel_gauss = 1 / 16 * np.array([[1, 2, 1],
                                  [2, 4, 2],
                                  [1, 2, 1]])

img_filter_RGB = cv2.filter2D(imgRGB, -1, kernel)
img_box_RGB = cv2.filter2D(imgRGB, -1, kernel_box)
img_gauss_RGB = cv2.filter2D(imgRGB, -1, kernel_gauss)
img_another_RGB = cv2.filter2D(imgRGB, -1, kernel_another)
img_median_RGB = cv2.medianBlur(imgRGB, 3)


cv2.imshow("Original", imgRGB)
cv2.imshow("Filter", img_filter_RGB)
cv2.imshow("Another filter", img_another_RGB)
cv2.imshow("Gauss filter", img_gauss_RGB)
cv2.imshow("Median filter", img_median_RGB)
cv2.imshow("Box filter", img_box_RGB)

cv2.waitKey(0)
cv2.destroyAllWindows()
