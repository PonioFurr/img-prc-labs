import cv2
import numpy as np
import math
import os
import glob
import json

text_font = cv2.FONT_HERSHEY_SIMPLEX

color = {"black": (0, 0, 0),
         "white": (255, 255, 255),
         "red": (0, 0, 255)}


def __clamp(value, bounds):
    if value > bounds[1]:
        return bounds[1]
    elif value < bounds[0]:
        return bounds[0]
    else:
        return value


# Приведение насыщенности и яркости вклеимового знака к аналогичным характеристикам исходного
def __color_adjustment(image_src, image_dst, position):
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
            image_dst_hsv[i, j, 1] = __clamp(image_dst_hsv[i, j, 1] * value_k[0], (0, 255))
            image_dst_hsv[i, j, 2] = __clamp(image_dst_hsv[i, j, 2] * value_k[1], (0, 255))

    image_result = cv2.cvtColor(image_dst_hsv, cv2.COLOR_HSV2BGR)

    b_channel, g_channel, r_channel = cv2.split(image_result)
    _, _, _, a_channel = cv2.split(image_dst)

    gauss_w = 13
    a_channel = cv2.GaussianBlur(a_channel, (gauss_w, gauss_w), 0)
    #cv2.imshow("", a_channel)


    return cv2.merge((b_channel, g_channel, r_channel, a_channel))


# Вклеивание одного изображения поверх другого с учётом альфа канала
def __overlay_image(image, image_overlay, position, image_alpha):
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


def __draw_signs(image, objects, sign_list, sign_images_folder, debug):
    for obj in objects:
        if obj["category"] in sign_list.keys():     # Если такой знак описан в "signs"
            offset = 16                             # Небольшой отступ, чтобы при трансформации не срезало края
            obj_location = (int(obj["bbox"]["xmin"]) - offset,  # Позиция bbox сдвинутого на offset влево
                            int(obj["bbox"]["ymin"]) - offset)
            obj_bbox = (int(obj["bbox"]["xmax"] - obj_location[0] + offset),    # Габариты bbox, расширенного на offset
                        int(obj["bbox"]["ymax"] - obj_location[1] + offset))

            sign_shape = sign_list[obj["category"]]["shape"]    # Тип заменяемого знака polygon или ellipse

            sign_points = sign_list[obj["category"]]["points"]  # Точки заменяемого знака

            image_sign = cv2.imread(sign_images_folder +       # Исходная картинка вклеиваемого знака
                                    sign_list[obj["category"]]["image_name"], cv2.IMREAD_UNCHANGED)

            image_overlay = np.array([])                        # Вклеиваемый знак

            points_source = np.float32(sign_points)             # Точки знака на исходной картинке вклеиваемого знака

            #   Проверка на наличие нужных для вклеивания данных
            if image_sign is not None and ((sign_shape == 'ellipse' and "ellipse" in obj) or
                                           (sign_shape == 'polygon' and "polygon" in obj and
                                            len(sign_points) == len(points_source))):
                #   Расчёт координат места, куда будет вклеен знак
                if sign_shape == "ellipse":
                    ellipse_center = (obj["ellipse"][0][1], obj["ellipse"][0][0])
                    ellipse_axis = (obj["ellipse"][1][1] / 2, obj["ellipse"][1][0] / 2)
                    angle = math.radians(obj["ellipse"][2])
                    points_destination = np.float32([[ellipse_center[1], ellipse_center[0]] for _ in range(4)]) + \
                                         np.float32([[ellipse_axis[i % 2] * -math.sin(-angle + math.pi / 2 * i),
                                                      ellipse_axis[i % 2] * -math.cos(-angle + math.pi / 2 * i)]
                                                     for i in range(4)]) - np.float32([obj_location])
                elif sign_shape == "polygon":
                    points_destination = np.float32(obj["polygon"]) - \
                                         np.float32(obj_location)

                    if len(points_source) > 4:
                        combination_num = len(points_source) // 4
                        _points_source = list()
                        _points_destination = list()

                        for i in range(3):
                            for j in range(combination_num):
                                coord_sum_dst = [0, 0]
                                coord_sum_dst[0] += points_destination[i * combination_num + j][0]
                                coord_sum_dst[1] += points_destination[i * combination_num + j][1]

                                coord_sum_src = [0, 0]
                                coord_sum_src[0] += points_source[i * combination_num + j][0]
                                coord_sum_src[1] += points_source[i * combination_num + j][1]

                            _points_destination.append([coord_sum_dst[0] / combination_num,
                                                        coord_sum_dst[1] / combination_num])
                            _points_source.append([coord_sum_src[0] / combination_num,
                                                   coord_sum_src[1] / combination_num])

                        for j in range(len(points_source) - 3 * combination_num):
                            coord_sum_dst = [0, 0]
                            coord_sum_dst[0] += points_destination[3 * combination_num + j][0]
                            coord_sum_dst[1] += points_destination[3 * combination_num + j][1]

                            coord_sum_src = [0, 0]
                            coord_sum_src[0] += points_source[3 * combination_num + j][0]
                            coord_sum_src[1] += points_source[3 * combination_num + j][1]

                        _points_destination.append([coord_sum_dst[0] / combination_num,
                                                   coord_sum_dst[1] / combination_num])
                        _points_source.append([coord_sum_src[0] / combination_num,
                                                   coord_sum_src[1] / combination_num])

                        points_destination = np.float32(_points_destination)
                        points_source = np.float32(_points_source)

                #   Трансформация исходного изображения знака
                #   Для эллипса выполняются доп. преобразования поворота содержимого знака
                if sign_shape == "ellipse":
                    t_matrix = cv2.getRotationMatrix2D((image_sign.shape[0] // 2, image_sign.shape[1] // 2),
                                                       math.degrees(angle), 1)
                    image_overlay = cv2.warpAffine(image_sign, t_matrix,
                                                   (image_sign.shape[0], image_sign.shape[1]))

                    t_matrix = cv2.getPerspectiveTransform(points_source, points_destination)
                    image_overlay = cv2.warpPerspective(image_overlay, t_matrix, obj_bbox)
                elif sign_shape == "polygon":
                    if len(points_source) == 3:
                        t_matrix = cv2.getAffineTransform(points_source, points_destination)
                        image_overlay = cv2.warpAffine(image_sign, t_matrix, obj_bbox)
                    elif len(points_source) == 4:
                        t_matrix = cv2.getPerspectiveTransform(points_source[:4], points_destination[:4])
                        image_overlay = cv2.warpPerspective(image_sign, t_matrix, obj_bbox)

                #   Обработка вклеимового знака (цвет и размытие)
                if image_overlay != np.array([]):
                    image_overlay = __color_adjustment(image, image_overlay, (obj_location[0], obj_location[1]))

                    cv2.boxFilter(image_overlay, -1, (3, 3))

                    #   Вклеивание знака
                    __overlay_image(image, image_overlay, obj_location, 1)
                    #   Обрамление заменённого знака для теста
                    if debug:
                        cv2.rectangle(image, obj_location,
                                      (obj_location[0] + obj_bbox[0], obj_location[1] + obj_bbox[1]), color["red"], 5)


def process(task_file,
            input_images_folder,
            signs_description_folder,
            sign_images_folder,
            output_folder,
            show_result=False,
            signs_description_prefix='sign_'):


    #   Парсинг JSON файлов
    #   Подгрузка инфы по знакам
    try:
        source_list = json.load(open(task_file))["imgs"]
    except:
        print("Can't open {}".format(task_file))
        exit()

    sign_list = dict()
    for file in glob.glob(os.path.join(signs_description_folder, "{}*.json".format(signs_description_prefix))):
        try:
            _sign_annot = json.load(open(file))
            sign_list = {**sign_list, **_sign_annot}
        except:
            print("Can't open {}".format(file))

    #   Обработка каждой картинки, описаной в файле "annotations"
    for img_id in list(source_list.keys()):
        process_object = source_list[img_id]
        path = os.path.join(input_images_folder, str(process_object["path"]))
        if os.path.isfile(path):
            imgRGB = cv2.imread(path)

            #   Для каждого знака
            __draw_signs(imgRGB, source_list[img_id]["objects"], sign_list, sign_images_folder, False)

            cv2.imwrite(os.path.join(output_folder, str(process_object["id"]) + '.jpg'), imgRGB)

            if show_result:
                k = imgRGB.shape[0] // imgRGB.shape[1]
                cv2.imshow("Output" + str(img_id), cv2.resize(imgRGB, (900 * k, 900)))
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        else:
            print("No file: " + path)