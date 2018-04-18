import json
import cv2
import os.path
import numpy as np
import math

annotations_file = "D:\\dataset\\annotations.json"
source_images_folder = "D:\\dataset\\"
output_images_folder = "D:\\dataset\\output\\"
signs_annotation_file = "D:\\dataset\\russian_signs\\signs.json"
signs_images_folder = "D:\\dataset\\russian_signs\\"

text_font = cv2.FONT_HERSHEY_SIMPLEX

color = {"black": (0, 0, 0),
         "white": (255, 255, 255),
         "red": (0, 0, 255)}

sign_db = dict()

#   Sign Shapes
normalized_shape = dict()
normalized_shape["triangle_direct"] = [[0.5, 0.0], [0.0, 1.0], [1.0, 1.0]]
normalized_shape["triangle_reversed"] = [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]
normalized_shape["ellipse"] = [[1.0, 0.5], [0.5, 1.0], [0, 0.5], [0.5, 0]]
normalized_shape["diamond"] = [[1.0, 0.5], [0.5, 0], [0, 0.5], [0.5, 1.0]]


#   JSON file parsing
source_list = json.load(open(annotations_file))["imgs"]
sign_list = json.load(open(signs_annotation_file))
print(sign_list)


def clamp(value, bounds):
    if value > bounds[1]:
        return bounds[1]
    elif value < bounds[0]:
        return bounds[0]
    else:
        return value


def color_adjustment(image_src, image_dst, position):
    image_src_hsv = cv2.cvtColor(image_src, cv2.COLOR_BGR2HSV)
    image_dst_hsv = cv2.cvtColor(image_dst, cv2.COLOR_BGR2HSV)
    value_src = [0, 0]
    value_dst = [0, 0]

    for i in range(image_dst.shape[0]):
        for j in range(image_dst.shape[1]):
            alpha = image_dst[i, j, 3] / 255

            if i + position[1] > -1 and i + position[1] < image_src.shape[0] and \
                    j + position[0] > -1 and j + position[0] < image_src.shape[1]:
                value_src[0] += image_src_hsv[i + position[1], j + position[0], 1] * alpha
                value_src[1] += image_src_hsv[i + position[1], j + position[0], 2] * alpha
                value_dst[0] += image_dst_hsv[i, j, 1] * alpha
                value_dst[1] += image_dst_hsv[i, j, 2] * alpha

    value_k = (value_src[0] / value_dst[0], value_src[1] / value_dst[1])

    for i in range(image_dst_hsv.shape[0]):
        for j in range(image_dst_hsv.shape[1]):
            image_dst_hsv[i, j, 1] = clamp(image_dst_hsv[i, j, 1] * value_k[0], (0, 255))
            image_dst_hsv[i, j, 2] = clamp(image_dst_hsv[i, j, 2] * value_k[1], (0, 255))

    image_result = cv2.cvtColor(image_dst_hsv, cv2.COLOR_HSV2BGR)

    b_channel, g_channel, r_channel = cv2.split(image_result)
    _, _, _, a_channel = cv2.split(image_dst)

    return cv2.merge((b_channel, g_channel, r_channel, a_channel))


def overlay_image(image, image_overlay, position, image_alpha):
    for i in range(image_overlay.shape[0]):
        for j in range(image_overlay.shape[1]):
            if i + position[1] > -1 and i + position[1] < image.shape[0] and \
                j + position[0] > -1 and j + position[0] < image.shape[1]:
                alpha = image_overlay[i, j, 3] / 255 * image_alpha
                color = image[position[1] + i, position[0] + j]
                color_overlay = image_overlay[i, j]
                image[position[1] + i, position[0] + j] = [color[0] * (1 - alpha) + color_overlay[0] * alpha,
                                                           color[1] * (1 - alpha) + color_overlay[1] * alpha,
                                                           color[2] * (1 - alpha) + color_overlay[2] * alpha]


def draw_signs(image, objects, debug):
    for obj in objects:
        if obj["category"] in sign_list.keys():   # Если знак есть в БД
            offset = 16     # Небольшой сдвиг, чтобы при трансформации не срезало края
            obj_location = (int(obj["bbox"]["xmin"]) - offset,
                            int(obj["bbox"]["ymin"]) - offset)

            obj_bbox = (int(obj["bbox"]["xmax"] - obj_location[0] + offset),
                        int(obj["bbox"]["ymax"] - obj_location[1] + offset))

            sign_shape = sign_list[obj["category"]]["shape"]

            image_sign = cv2.imread(signs_images_folder +
                                    sign_list[obj["category"]]["image_name"], cv2.IMREAD_UNCHANGED)

            cv2.imshow(str(image_sign),image_sign)

            #   В зависимости от формы знака
            img_overlay = np.array([])

            if sign_shape == "triangle_direct" and "polygon" in list(obj.keys()) and len(obj["polygon"]) == 3:
                points_source = np.float32(normalized_shape["triangle_direct"]) * \
                                np.float32([image_sign.shape[1], image_sign.shape[0]])

                points_destination = np.float32([obj["polygon"]]) - \
                                     np.float32([obj_location])

                t_matrix = cv2.getAffineTransform(points_source, points_destination)
                img_overlay = cv2.warpAffine(image_sign, t_matrix, obj_bbox)

            elif sign_shape == "ellipse" and "ellipse" in list(obj.keys()) and len(obj["ellipse"]) == 3:
                    points_source = np.float32(normalized_shape["ellipse"]) * \
                                    np.float32([image_sign.shape[1], image_sign.shape[0]])

                    ellipse_center = (obj["ellipse"][0][1], obj["ellipse"][0][0])
                    ellipse_axis = (obj["ellipse"][1][1] / 2, obj["ellipse"][1][0] / 2)
                    angle = math.radians(obj["ellipse"][2])

                    points_destination = np.float32([[ellipse_center[1], ellipse_center[0]] for _ in range(4)]) + \
                                         np.float32([[ellipse_axis[(i+1) % 2] * math.cos(angle + math.pi / 2 * i),
                                                      ellipse_axis[(i+1) % 2] * math.sin(angle + math.pi / 2 * i)]
                                                     for i in range(4)]) - np.float32([obj_location])

                    t_matrix = cv2.getRotationMatrix2D((image_sign.shape[0] // 2, image_sign.shape[1] // 2),
                                                       math.degrees(angle), 1)
                    img_overlay = cv2.warpAffine(image_sign, t_matrix,
                                                 (image_sign.shape[0], image_sign.shape[1]))

                    t_matrix = cv2.getPerspectiveTransform(points_source, points_destination)
                    img_overlay = cv2.warpPerspective(img_overlay, t_matrix, obj_bbox)
            else:
                if "polygon" in obj.keys():
                    print(obj["polygon"])

            if img_overlay != np.array([]):
                method = 0
                img_overlay = color_adjustment(image, img_overlay, (obj_location[0], obj_location[1]))

                if method == 0:
                    cv2.boxFilter(img_overlay, -1, (3, 3))
                elif method == 1:
                    kernel_gauss = 1 / 16 * np.array([[1, 2, 1],
                                                      [2, 4, 2],
                                                      [1, 2, 1]])
                    img_overlay = cv2.filter2D(img_overlay, -1, kernel_gauss)

                overlay_image(image, img_overlay, obj_location, 1)

                if debug:
                    cv2.rectangle(image, obj_location,
                                (obj_location[0] + obj_bbox[0], obj_location[1] + obj_bbox[1]),
                                color["red"], 5)


for id in list(source_list.keys()):
    object = source_list[id]
    path = source_images_folder + str(object["id"]) + ".jpg"

    if os.path.isfile(path):
        imgRGB = cv2.imread(path)
        #   Для каждого знака
        draw_signs(imgRGB, source_list[id]["objects"], False)

        k = imgRGB.shape[0] // imgRGB.shape[1]
        cv2.imshow("Image" + str(id), cv2.resize(imgRGB, (900 * k, 900)))
        cv2.imwrite(output_images_folder + str(id) + ".jpg", imgRGB)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print("No file: " + path)

