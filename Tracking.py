from statistics import mean

import numpy as np
import cv2

class Tracking:
    detected_area = 400
    detected_distance = 50
    frame_count = 0
    object_id = 0
    id = 0
    n_dominant_colors = 2
    frame_h = 0
    frame_w = 0

    def __init__(self):
        self.boxes = []
        self.objects = []
        self.deleted_objects = []
        self.object_id = 0

    def reset_boxes(self):
        self.boxes = []

    def add_box(self, x, y, w, h):
        self.boxes.append({"x": x, "y": y, "w": w, "h": h, "cx": x + w // 2, "cy": y + h // 2})

    def add_object(self, object_id, coords):
        self.objects.append(
            {"object_id": object_id, "checked_id": True if len(self.deleted_objects) == 0 else False, "x": coords["x"], "y": coords["y"], "w": coords["w"], "h": coords["h"],
             "cx": coords["cx"],  "cy": coords["cy"], "color_data": []
             })
        object_id += 1

    def add_color_data(self, object_id, color_data):
        for obj in self.objects:
            if obj["object_id"] == object_id:
                obj["color_data"].append(color_data)

    def remove_object(self, ob):
        for i, obj in enumerate(self.objects):
            if obj["object_id"] == ob["object_id"]:
                t_max = []
                for o in obj["color_data"]:
                    t_max.append(o["height"])
                if len(t_max) > 0:
                    # print("t_max", max(t_max), t_max.index(max(t_max)))
                    obj["color_data"] = obj["color_data"][t_max.index(max(t_max))]
                if len(obj["color_data"]) > 0:
                    self.deleted_objects.append(obj)
                del self.objects[i]
                return

    def update_object(self, object_id, ob):
        for obj in self.objects:
            if obj["object_id"] == object_id:
                obj["x"] = ob["x"]
                obj["y"] = ob["y"]
                obj["w"] = ob["w"]
                obj["h"] = ob["h"]
                obj["cx"] = ob["cx"]
                obj["cy"] = ob["cy"]

                # Вычисляем цветовой вес текущего объекта oтносительго удаленных (color_weight)
                # Для этого в максимально близких цветах суммируем разницу этих цветов по 3 каналам - RGB
                # И делим полученное значение на количество доментных цветов (n_dominant_colors)
                # Средняя разница в процентах по одному каналу = color_weight / n_dominant_colors/ 3 (RGB) * (100% / 256 (размерность канала))

                # есть удаленные объекты для сопоставления цвета
                if not obj["checked_id"] and len(self.deleted_objects) > 0 and len(obj["color_data"]) > 0:
                    # проверяем зашел ли объект полностью в кадр
                    if obj["y"] > 0 and obj["y"] + obj["h"] < self.frame_h:
                        if 'color' in obj["color_data"][0]:
                            obj_colors = obj["color_data"][0]['color']
                            #print("OB", obj_colors)
                            color_weight = []
                            for do in self.deleted_objects:
                                sum_w = 0
                                for d_colors in do["color_data"]['color']:
                                    c_w = []
                                    for o_colors in obj_colors:
                                        c_w.append(abs(d_colors[0] - o_colors[0]) + abs(d_colors[1] - o_colors[1]) + abs(d_colors[2] - o_colors[2]))
                                    sum_w += min(c_w)
                                color_weight.append(sum_w/self.n_dominant_colors/3*(100/256))
                            print("color_weigh", color_weight)
                            # Близость объекта в процентах
                            if min(color_weight) < 10:
                                idx = color_weight.index(min(color_weight))
                                if idx < len(self.deleted_objects):
                                    obj["checked_id"] = True
                                    obj["object_id"] = self.deleted_objects[idx]["object_id"]
                                    del self.deleted_objects[idx]
                                    return
                        obj["checked_id"] = True

    # Доминантная цветовая палитра объекта в рамке
    @staticmethod
    def visualize_colors(cluster, centroids, height):
        color_data = {"color": [], "percent": []}
        # Get the number of different clusters, create histogram, and normalize
        labels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
        (hist, _) = np.histogram(cluster.labels_, bins=labels)
        hist = hist.astype("float")
        hist /= hist.sum()

        color_data = {"height": height, "color": centroids, "percent": hist}

        # Вариант сортировки по цвету
        tmp = sorted([(percent, color, color.sum()) for (percent, color) in zip(hist, centroids)], key=lambda x: x[2])
        #print(tmp)
        # Create frequency rect and iterate through each cluster's color and percentageq
        rect = np.zeros((30, 300, 3), dtype=np.uint8)
        # Вариант сортировки по процентам
        # colors = sorted([(percent, color) for (percent, color) in zip(hist, centroids)])
        start = 0
        for (percent, color, _) in tmp:
            # print(color, "{:0.1f}%".format(percent * 100))
            end = start + (percent * 300)
            cv2.rectangle(rect, (int(start), 0), (int(end), 50), \
                          color.astype("uint8").tolist(), -1)
            start = end
        return rect, color_data

