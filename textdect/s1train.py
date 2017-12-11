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
import json
import os

from textdect.yolonet import YoloNet


def train(config_file, train_image_dir, train_label_file,  weights_path_file, train_log_dir):
    with open(config_file) as config_buffer:
        config = json.loads(config_buffer.read())

    file_list = []
    for img_file in os.listdir(train_image_dir):
        full_path_name = os.path.join(img_dir, img_file)
        if os.path.isfile(full_path_name) and img_file.lower().endswith(tuple(['.jpg', '.png'])):
            file_list.append(img_file)

    train_data = read_image_data(config, file_list, train_label_file)
    print("# Training data size: {}".format(len(train_data)))

    yolo_net = YoloNet(config)
    yolo_net.train(train_image_dir, train_data,  weights_path_file, train_log_dir)


def read_image_data(config, image_file_list, label_path_file):
    """
    A sample line in the label file:
        aa.jpg; [100, 120, 200, 156]; [100, 200, 208, 248]
    """
    image_data = []

    VALID_MIN_X = config['image_left_skip']
    VALID_MAX_X = 640 - config['image_right_skip']

    with open(label_path_file, 'r') as label_f:
        for line in label_f:
            ln = line.strip()
            if not ln:
                continue
            img_item = {}
            s = ln.split(';')
            if len(s) == 1 or s[0].strip() not in image_file_list:
                continue
            img_item['filename'] = s[0].strip()

            tmp = []
            OUT_OF_RANGE = False
            for i in range(1, len(s)):
                xmin, ymin, xmax, ymax = s[i].strip()[1:-1].split(',')
                tmp.append((xmin, ymin, xmax, ymax))
                if int(xmin) < VALID_MIN_X or int(xmax) > VALID_MAX_X:
                    OUT_OF_RANGE = True

            img_item['labels'] = tmp
            if not OUT_OF_RANGE:
                image_data.append(img_item)
            else:
                print("# At least one box in image {} is out of range. Sample skiped."
                      .format(img_item['filename']))

    return image_data


if __name__ == '__main__':
    from settings import PROJECT_ROOT

    img_dir = os.path.join(PROJECT_ROOT, 'Data', 'Step1', 'Training', 'AntImages')
    label_file = os.path.join(PROJECT_ROOT, 'Data', 'Step1', 'Training', 'labels.txt')

    log_dir = os.path.join(PROJECT_ROOT, 'Data', 'Result', 'Logs')

    weights_file = os.path.join(PROJECT_ROOT, 'Data', 'Result', 's1_model_weights.{epoch:02d}.h5')

    train('config.json', img_dir, label_file,  weights_file, log_dir)
