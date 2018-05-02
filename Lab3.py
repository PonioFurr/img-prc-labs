import cv2
import numpy as np
import math

def clamp(value, bounds):
    if value > bounds[1]:
        return bounds[1]
    elif value < bounds[0]:
        return bounds[0]
    else:
        return value

def hist_image(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    max = hist[0]
    for i in range(hist.shape[0]):
        if hist[i] > max:
            max = hist[i]

    histogram_img = np.zeros((255, 255), np.uint8)
    for i in range(histogram_img.shape[0]):
        value = int(255 * (hist[i] / max)) - 1
        cv2.line(histogram_img, (i, 255), (i, 255 - value), 255)

    steps = 20
    for i in range(20):
        cv2.line(histogram_img, (i * 255 // steps, 0), (i * 255 // steps, 5), 255)

    return histogram_img

threshold = (0.0, 0.85)

imgGRY = cv2.cvtColor(cv2.imread("images/test3.jpg"), cv2.COLOR_BGR2GRAY)
img_result_GRY = np.copy(imgGRY)

for i in range(img_result_GRY.shape[0]):
    for j in range(img_result_GRY.shape[1]):
        result = img_result_GRY[i, j] - threshold[0] * 255
        result *= 1 / (1 - threshold[0] - (1 - threshold[1]))
        img_result_GRY[i, j] = clamp(result, (0, 255))

cv2.imwrite('renders/Lab3.png', img_result_GRY)
cv2.imwrite('renders/Lab3_histogram_before.png', hist_image(imgGRY))
cv2.imwrite('renders/Lab3_histogram_after.png', hist_image(img_result_GRY))

cv2.imshow("Histogram before", hist_image(imgGRY))
cv2.imshow("Histogram after", hist_image(img_result_GRY))
cv2.imshow("Original", imgGRY)
cv2.imshow("Result", img_result_GRY)

cv2.waitKey(0)
cv2.destroyAllWindows()