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
import cv2
import numpy as np
import os

from tensorflow import keras


class BatchGenerator(keras.utils.Sequence):
    def __init__(self, image_dir, image_data, config):
        self.generator = None

        self.image_dir = image_dir
        self.image_data = image_data
        self.config = config

        np.random.shuffle(self.image_data)

    def __len__(self):
        return int(np.ceil(float(len(self.image_data))/self.config['batch_size']))

    def __getitem__(self, idx):
        l_bound = idx * self.config['batch_size']
        r_bound = (idx+1) * self.config['batch_size']
        
        if r_bound > len(self.image_data):
            r_bound = len(self.image_data)
            l_bound = r_bound - self.config['batch_size']

        # Initialize x_batch and y_batch
        x_batch = np.zeros((r_bound - l_bound, self.config['image_height'], self.config['image_width'], 3))
        y_batch = np.zeros((r_bound - l_bound, self.config['grid_y_count'], self.config['grid_x_count'], 4+1))

        instance_idx = 0
        for img_item in self.image_data[l_bound:r_bound]:
            img = cv2.imread(os.path.join(self.image_dir, img_item['filename']))
            img = img[:, self.config['image_left_skip']:-self.config['image_right_skip'], ::-1]

            x_batch[instance_idx] = self.normalize(img)

            for xmin, ymin, xmax, ymax in img_item['labels']:
                xmin, ymin = int(xmin) - self.config['image_left_skip'], int(ymin)
                xmax, ymax = int(xmax) - self.config['image_left_skip'], int(ymax)
                if xmax > xmin and ymax > ymin:
                    center_x = 0.5 * (xmin + xmax)
                    center_x = center_x / self.config['grid_x_size']
                    center_y = 0.5 * (ymin + ymax)
                    center_y = center_y / self.config['grid_y_size']

                    grid_x_idx = int(np.floor(center_x))
                    grid_y_idx = int(np.floor(center_y))

                    if grid_x_idx < self.config['grid_x_count'] and grid_y_idx < self.config['grid_y_count']:
                        box_w = (xmax - xmin) / self.config['grid_x_size']
                        box_h = (ymax - ymin) / self.config['grid_y_size']
                        
                        box = [center_x - grid_x_idx, center_y - grid_y_idx, box_w, box_h]
                                
                        # Assign ground truth x, y, w, h, and confidence to y_batch
                        y_batch[instance_idx, grid_y_idx, grid_x_idx, 0:4] = box
                        y_batch[instance_idx, grid_y_idx, grid_x_idx, 4] = 1.

            # Increase instance counter in current batch
            instance_idx += 1

        return x_batch, y_batch

    def on_epoch_end(self):
        np.random.shuffle(self.image_data)

    @staticmethod
    def normalize(image):
        return image / 255.
