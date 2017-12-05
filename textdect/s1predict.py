# Copyright 2017 Bo Shao. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
This script is based on the original Keras model, and serves as a test/validation tool during
the model development. It is not used as a final product in part of the pipeline.
"""
import cv2
import json

from textdect.yolonet import YoloNet


def predict(config_file, weights_path_file, predict_file_list, out_dir):
    with open(config_file) as config_buffer:
        config = json.loads(config_buffer.read())

    yolo_net = YoloNet(config)
    yolo_net.load_weights(weights_path_file)

    for img_file in predict_file_list:
        image = cv2.imread(img_file)
        boxes = yolo_net.predict(image)
        image = draw_boxes(image, boxes)

        _, filename = os.path.split(img_file)
        cv2.imwrite(os.path.join(out_dir, filename), image)


def draw_boxes(image, boxes):
    # print("box len = {}".format(len(boxes)))
    for box in boxes:
        xmin, ymin, xmax, ymax = box.get_coordinates()
        # print("{}, {}, {}, {}".format(xmin, ymin, xmax, ymax))
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)

    return image


if __name__ == '__main__':
    import os
    from settings import PROJECT_ROOT

    weights_file = os.path.join(PROJECT_ROOT, 'Data', 'Result', 's1_model_weights.h5')
    img_dir = os.path.join(PROJECT_ROOT, 'Data', 'OtmImages')
    out_dir = os.path.join(PROJECT_ROOT, 'Data', 'Temp')

    file_list = []
    file_count = 0
    for img_file in sorted(os.listdir(img_dir)):
        full_path_name = os.path.join(img_dir, img_file)
        if os.path.isfile(full_path_name) and img_file.lower().endswith(tuple(['.jpg', '.png'])):
            file_count += 1
            if file_count > 0:
                file_list.append(full_path_name)
                if file_count >= 200:
                    break

    predict('config.json', weights_file, file_list, out_dir)