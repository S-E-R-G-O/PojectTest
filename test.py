import math
from statistics import median
from sklearn.cluster import KMeans
import cv2
import numpy as np
from PIL import Image
from Tracking import Tracking

# Создание объекта фона
mog = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
# Открытие видео
stream1 = cv2.VideoCapture('3.Camera 2017-05-29 16-23-04_137 [3m3s].avi')
stream2 = cv2.VideoCapture('4.Camera 2017-05-29 16-23-04_137 [3m3s].avi')
stream = cv2.VideoCapture('Cars.mp4')


# Initialize
tracking = Tracking()
boxes_prev = []
count = 0

while True:
    # Захват кадров из обоих видео
    #ret1, frame1 = stream1.read()
    #ret2, frame2 = stream2.read()
    ret, frame = stream.read()

    if not ret:
        break

    Tracking.frame_count += 1

    frame_res = cv2.resize(frame, (int(frame.shape[1]/2),int(frame.shape[0]/2)))
    Tracking.frame_h, Tracking.frame_w = frame.shape[:2]

    mask = mog.apply(frame_res)
    _, thresh = cv2.threshold(mask, 180, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_eroded = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    thresh = cv2.medianBlur(thresh, 15)
    thresh = cv2.dilate(thresh, None, iterations=4)

    cntr, hirarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    tracking.reset_boxes()

    if len(cntr) != 0:
        # Найти самый большой контур
        max_cnt = max(cntr, key=cv2.contourArea)
        max_area = cv2.contourArea(max_cnt)

        if max_area > Tracking.detected_area:
            count += 1

        # Если в кадере есть отдетектированные контуры - проходим по ним всем
        for cnt in cntr:
            area = cv2.contourArea(cnt)
            if area > Tracking.detected_area and area > max_area * 0.5:
                x, y, w, h = cv2.boundingRect(cnt)
                cx = x + w // 2
                cy = y + h // 2
                tracking.add_box(x, y, w, h)
                print("count", len(tracking.boxes))

        # Если мы вначале детекции - на шаге 1-2
        if count <= 2:
            for cur in tracking.boxes:
                for prev in boxes_prev:
                    distance = math.hypot(prev["cx"] - cur["cx"], prev["cy"] - cur["cy"])

                    if distance > Tracking.detected_distance:
                        tracking.add_object(tracking.id, cur)
                        tracking.id += 1

        # На шагах 3, 4 ...
        else:
            tracking_objects_copy = tracking.objects.copy()
            boxes_copy = tracking.boxes.copy()

            for obj in tracking_objects_copy:
                object_exists = False
                for box in boxes_copy:
                    distance = math.hypot(obj["cx"] - box["cx"], obj["cy"] - box["cy"])
                    #print("distance", distance)

                    # Update IDs position
                    if distance < Tracking.detected_distance:
                        # предыдущий объект близок к теущему местоположению центра бокса
                        tracking.update_object(obj["object_id"], box)
                        object_exists = True

                        if box in tracking.boxes:
                            tracking.boxes.remove(box)
                        continue

                # Удаляем не отслеживаемый объект и записываем данные о его доминатном цвете в Tracking.deleted_objects
                if not object_exists:
                    # print("DEL", obj["object_id"])
                    tracking.remove_object(obj)

            # Add new IDs found
            for box in tracking.boxes:
                tracking.add_object(tracking.id, box)
                # print("ADD", tracking.objects)
                tracking.id += 1

        # Показать все детектированые в текущем фрейме объекты с центром и ID
        if max_area > Tracking.detected_area:
            for obj in tracking.objects:
                _cx = obj["cx"]
                _cy = obj["cy"]
                _x = obj["x"]
                _y = obj["y"]
                _h = obj["h"]
                _w = obj["w"]
                #в if - задаем габариты ширины и высоты, при которм рисуем рамки
                if _w > 50 and _h > 100:
                    try:
                        image_in_box = frame_res[_y:_y + _h, _x:_x + _w]
                        mask_in_box = thresh[_y:_y + _h, _x:_x + _w]
                        # cv2.imshow('mask_in_box', mask_in_box)
                        #b, g, r = cv2.split(image_in_box)
                        #gray = cv2.cvtColor(image_in_box, cv2.COLOR_BGR2GRAY)
                        #ret, mask_all = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                        # masked = cv2.bitwise_and(opencv_image_all, opencv_image_all, mask=thresh1)
                        # cv2.imshow('Mask1', masked)
                        m_h, m_w, channels = image_in_box.shape

                        opencv_mask_1 = mask_in_box[0:m_h // 2, 0:m_w]
                        opencv_image_1 = image_in_box[0:m_h // 2, 0:m_w]
                        #cv2.imshow('Obj1', opencv_mask_1)
                        opencv_mask_2 = mask_in_box[m_h // 2:m_h, 0:m_w]
                        opencv_image_2 = image_in_box[m_h // 2:m_h, 0:m_w]
                        #cv2.imshow('Obj2', opencv_image_2)

                        # ВЕРХ -- Среднее заначения для каждого из 3 цветов BGR
                        # Гистограммы - вдруг понадобится
                        # hist_B = cv2.calcHist([opencv_image_1], [0], None, [256], [0, 256])
                        # hist_G = cv2.calcHist([opencv_image_1], [1], None, [256], [0, 256])
                        # hist_R = cv2.calcHist([opencv_image_1], [2], None, [256], [0, 256])
                        mean = cv2.mean(opencv_image_1, mask=opencv_mask_1)
                        blue_mean = mean[0]
                        green_mean = mean[1]
                        red_mean = mean[2]

                        #average = opencv_image_1.mean(axis=0).mean(axis=0)
                        b, g, r = round(blue_mean), round(green_mean), round(red_mean)
                        # !!! color in BGR
                        # Fill
                        cv2.circle(frame_res, (22, 25), 20, (b, g, r), -1)
                        # Outline
                        cv2.circle(frame_res, (22, 25), 20, (200, 200, 200), 1)
                        # color in RGB
                        cv2.putText(frame_res, "TOP RGB: {} {} {}".format(r, g, b),
                                    (50, 30), 0, 0.5, (255, 225, 255), 2)

                        # НИЗ -- Среднее заначения для каждого из 3 цветов BGR
                        mean = cv2.mean(opencv_image_2, mask=opencv_mask_2)
                        blue_mean = mean[0]
                        green_mean = mean[1]
                        red_mean = mean[2]
                        # average = opencv_image_2.mean(axis=0).mean(axis=0)
                        (b, g, r) = (round(blue_mean), round(green_mean), round(red_mean))
                        # !!! color in BGR
                        # Fill
                        cv2.circle(frame_res, (22, 65), 20, (b, g, r), -1)
                        # Outline
                        cv2.circle(frame_res, (22, 65), 20, (200, 200, 200), 1)
                        # color in RGB
                        cv2.putText(frame_res, "BOT RGB: {} {} {}".format(r, g, b),
                                    (50, 70), 0, 0.5, (255, 225, 255), 2)

                        # Доминантная цветовая палитра объекта в рамке
                        image = cv2.cvtColor(image_in_box, cv2.COLOR_BGR2RGB)

                        reshape = image.reshape((image.shape[0] * image.shape[1], 3))
                        # n_clusters -  количество кластеров -> Доминатных цветов
                        cluster = KMeans(n_clusters=Tracking.n_dominant_colors).fit(reshape)
                        visualize, color_data = Tracking.visualize_colors(cluster, cluster.cluster_centers_, obj["h"])
                        # Запись доминатного цвета в объект
                        tracking.add_color_data(obj["object_id"], color_data)

                        visualize = cv2.cvtColor(visualize, cv2.COLOR_RGB2BGR)
                        # Отрисовка доминантных цветов после информации о усредненных цветах
                        x, y, h, w = 230, 15, visualize.shape[0], visualize.shape[1]
                        frame_res[y:y + h, x:x + w] = visualize
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 255, 0), 2)
                    except Exception as error:
                        print("An exception occurred:", error)

                    # Отрисовка информаци на видео - линии, боксы  и данные
                    cv2.rectangle(frame_res, (_x, _y), (_x + _w, _y + _h), (100, 255, 0), 2)
                    cv2.circle(frame_res, (_cx, _cy), 5, (0, 0, 255), -1)
                    cv2.putText(frame_res, str(obj["object_id"]) if obj["checked_id"] else "N/A", (_cx, _cy - 7), 0, 1,
                                (0, 0, 255), 2)

                    cv2.line(frame_res, (_x, _y + _h // 2), (_x + _w, _y + _h // 2),
                             (0, 0, 255), 1)

                    # cv2.line(frame, (obj["x"], obj["y"] + obj["h"] // 4), (obj["x"] + obj["w"], obj["y"] + obj["h"] // 4),(255, 0, 0), 1)
                    # cv2.line(frame, (obj["x"], int(obj["y"] + obj["h"] // 1.333)),(obj["x"] + obj["w"], int(obj["y"] + obj["h"] // 1.333)),(255, 0, 0), 1)

        if tracking.boxes:
            boxes_prev = tracking.boxes.copy()

    # Отображение результата

    cv2.imshow('Tracker', frame_res)
    cv2.imshow('Mask', thresh)

    # Make a copy of the points

    # Выход по нажатию 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        for do in tracking.deleted_objects:
            print("deleted object", do)
        break

# Освобождение ресурсов
stream1.release()
stream2.release()
cv2.destroyAllWindows()
