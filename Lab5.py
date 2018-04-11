import cv2
import numpy as np
import math
import random

color = (0, 255 / 2, 255)

ceil_xnum = 5
ceil_ynum = 2
ceil_w = 64

img_GRY = np.zeros((ceil_w * ceil_ynum, ceil_w * ceil_xnum, 1), np.uint8)
img_HSV = np.zeros((ceil_w * ceil_ynum, ceil_w * ceil_xnum, 3), np.uint8)

#   Strange pattern generation
for x in range(ceil_xnum):
    for y in range(ceil_ynum):
        bg_color_index = random.randint(0, len(color) - 1)
        while True:
            shape_color_index = random.randint(0, len(color) - 1)
            if shape_color_index != bg_color_index:
                break

        shape_type = random.choice(("circle", "rect", "triangle"))

        cv2.rectangle(img_GRY, (x * ceil_w, y * ceil_w), ((x + 1) * ceil_w, (y + 1) * ceil_w), (color[bg_color_index],), -1)
        if shape_type == "circle":
            cv2.circle(img_GRY, (int((x + 0.5) * ceil_w), int((y + 0.5) * ceil_w)), int(ceil_w / 3), (color[shape_color_index],), -1)
        elif shape_type == "rect":
            cv2.rectangle(img_GRY, (int((x + 0.2) * ceil_w), int((y + 0.2) * ceil_w)), (int((x + 0.8) * ceil_w), int((y + 0.8) * ceil_w)), (color[shape_color_index],), -1)
        elif shape_type == "triangle":
            points = np.array([[int((x + 0.2) * ceil_w), int((y + 0.8) * ceil_w)],
                               [int((x + 0.8) * ceil_w), int((y + 0.8) * ceil_w)],
                               [int((x + 0.5) * ceil_w), int((y + 0.2) * ceil_w)]], np.int32)
            points = points.reshape((3, 1, 2))
            cv2.fillPoly(img_GRY, [points], (color[shape_color_index],))
#   Calculation
for i in range(img_GRY.shape[0]):
    for j in range(img_GRY.shape[1]):
        img_HSV[i, j, 0] = (math.sqrt(((int(img_GRY[(i + 1) % img_HSV.shape[0], j, 0]) -
                                        int(img_GRY[i - 1, j, 0])) / 2) ** 2 +
                                      ((int(img_GRY[i, (j + 1) % img_HSV.shape[1], 0]) -
                                        int(img_GRY[i, j - 1, 0])) / 2) ** 2)) / 255 * 180
        img_HSV[i, j, 1] = (int(img_GRY[(i + 1) % img_HSV.shape[0], j, 0]) - int(img_GRY[i - 1, j, 0])) / 2
        img_HSV[i, j, 2] = (int(img_GRY[i, (j + 1) % img_HSV.shape[1], 0]) - int(img_GRY[i, j - 1, 0])) / 2


img_RGB = cv2.cvtColor(img_HSV, cv2.COLOR_HSV2BGR)

cv2.imshow("Original", img_GRY)
cv2.imshow("Result", img_RGB)

cv2.waitKey(0)
cv2.destroyAllWindows()