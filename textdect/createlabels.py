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
Other than using previously built sliding window version text detector to assist creating some 
training labels, we manually draw bounding boxes on images using MS paint (specify pure red as 
RGB = 255, 0, 0) with thickness = 1. Note that the image files have to be saved in PNG format. 
Otherwise, you will not be able to see the exact pixel values. Then run this script to create 
labels. Finally we use some other scripts to merge/move source image files and manually combine 
label texts.
"""
import cv2
import os
from collections import namedtuple


def find_horizontal_lines(image_array, threshold):
    line_dict = {}

    height, width, _ = image_array.shape
    for h in range(height):
        y, xmin, xmax = h, 0, 0
        for w in range(width):
            if image_array[h][w][0] == 0 and image_array[h][w][1] == 0 and image_array[h][w][2] == 255:
                if xmax == 0:
                    xmin, xmax = w, w
                elif w - xmax == 1:  # continuous
                    xmax = w
            elif xmax - xmin >= threshold:
                line_dict[HorLine(y=y, xmin=xmin, xmax=xmax)] = 1
                # print("Hor Line found: {}, {}, {}".format(y, xmin, xmax))
                break  # Do not search this line any more
            else: # interrupted
                xmin, xmax = 0, 0

    return line_dict


def find_vertical_lines(image_array, threshold):
    line_dict = {}

    height, width, _ = image_array.shape
    print("{}, {}".format(height, width))
    for w in range(width):
        x, ymin, ymax = w, 0, 0
        for h in range(height):
            if image_array[h][w][0] == 0 and image_array[h][w][1] == 0 and image_array[h][w][2] == 255:
                if ymax == 0:
                    ymin, ymax = h, h
                elif h - ymax == 1:  # continuous
                    ymax = h
            elif ymax - ymin >= threshold:
                line_dict[VerLine(x=x, ymin=ymin, ymax=ymax)] = 1
                # print("Ver Line found: {}, {}, {}".format(x, ymin, ymax))
                ymin, ymax = 0, 0  # Continue searching
            else:
                ymin, ymax = 0, 0

    return line_dict


def create_labels(image_dir, label_file):
    file_list = []
    for img_file in sorted(os.listdir(image_dir)):
        full_path_name = os.path.join(image_dir, img_file)
        if os.path.isfile(full_path_name) and img_file.lower().endswith('.png'):
            file_list.append(img_file)

    with open(label_file, 'a') as f_label:
        for img_file in file_list:
            full_path_name = os.path.join(image_dir, img_file)
            print(full_path_name)
            img_arr = cv2.imread(full_path_name)

            hor_dict = find_horizontal_lines(img_arr, threshold=20)
            ver_dict = find_vertical_lines(img_arr, threshold=12)

            box_list = []
            for (y, x1, x2), hv in hor_dict.items():
                # print("Hor Line = {}, {}, {}".format(y, x1, x2))
                for (x, y1, y2), vv in ver_dict.items():
                    # print("Ver Line = {}, {}, {}".format(x, y1, y2))
                    if x1 == x and y == y1:
                        xmin, ymin = x1, y1
                        if hor_dict[(y2, x1, x2)] == 1 and ver_dict[(x2, y1, y2)] == 1:
                            hor_dict[HorLine(y, x1, x2)] = 0
                            hor_dict[HorLine(y2, x1, x2)] = 0
                            ver_dict[VerLine(x, y1, y2)] = 0
                            ver_dict[VerLine(x2, y1, y2)] = 0

                            box_list.append(LabelBox(xmin, ymin, xmax=x2, ymax=y2))

            # The original image file is always in JPG format
            label_line = img_file[:-4] + ".jpg"
            for (xmin, ymin, xmax, ymax) in box_list:
                label_line += "; [{}, {}, {}, {}]".format(xmin, ymin, xmax, ymax)

            f_label.write("{}\n".format(label_line))


class HorLine(namedtuple("HorLine", ["y", "xmin", "xmax"])):
    pass


class VerLine(namedtuple("VerLine", ["x", "ymin", "ymax"])):
    pass


class LabelBox(namedtuple("LabelBox", ["xmin", "ymin", "xmax", "ymax"])):
    pass

if __name__ == '__main__':
    from settings import PROJECT_ROOT

    box_dir = os.path.join(PROJECT_ROOT, 'Data', 'Temp', 'BoxImages')
    label_file = os.path.join(PROJECT_ROOT, 'Data', 'Step1', 'label_manual.txt')
    create_labels(box_dir, label_file)
