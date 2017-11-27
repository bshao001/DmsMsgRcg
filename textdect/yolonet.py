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
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras

from textdect.batchgenerator import BatchGenerator
from textdect.yolomodel import TinyYolo, FullYolo


class YoloNet(object):
    def __init__(self, config):
        self.config = config

        self.image_height, self.image_width = config['image_height'], config['image_width']
        self.grid_h, self.grid_w = config['grid_y_count'], config['grid_x_count']

        if config['model_architecture'] == 'Full':
            self.model = FullYolo(self.image_height, self.image_width, self.grid_h, self.grid_w).model
        elif config['model_architecture'] == 'Tiny':
            self.model = TinyYolo(self.image_height, self.image_width, self.grid_h, self.grid_w).model
        else:
            raise Exception('Architecture not supported! Only Full Yolo and Tiny Yolo are supported '
                            'at the moment!')

        if config['debug']:
            self.model.summary()

    def train(self, image_dir, train_data, weights_path_file, log_dir):
        optimizer = keras.optimizers.Adam(lr=4e-4, beta_1=0.9, beta_2=0.999,
                                          epsilon=1e-08, decay=0.0)
        self.model.compile(loss=self.custom_loss, optimizer=optimizer)

        train_batch = BatchGenerator(image_dir, train_data, self.config)

        lr_schedule = keras.callbacks.LearningRateScheduler(self._schedule)
        checkpoint = keras.callbacks.ModelCheckpoint(weights_path_file,
                                                     monitor='loss',
                                                     verbose=1,
                                                     save_best_only=True,
                                                     mode='min',
                                                     period=1)
        tensorboard = keras.callbacks.TensorBoard(log_dir=log_dir,
                                                  histogram_freq=0,
                                                  write_graph=True,
                                                  write_images=False)

        self.model.fit_generator(generator=train_batch,
                                 steps_per_epoch=len(train_batch),
                                 epochs=self.config['num_epoch'],
                                 verbose=2,
                                 callbacks=[lr_schedule, checkpoint, tensorboard],
                                 workers=3,
                                 max_queue_size=8)

    def predict(self, image):
        input_image = self.normalize(image[:, self.config['image_left_skip']:-self.config['image_right_skip'],
                                     ::-1])
        input_image = np.expand_dims(input_image, 0)

        netout = self.model.predict(input_image)[0]
        boxes = self._decode_netout(netout)

        return boxes

    def load_weights(self, weight_path_file):
        self.model.load_weights(weight_path_file)

    def _schedule(self, epoch_num):
        if epoch_num > 0:
            print("# Starting epoch {:2d}, learning rate used in the last epoch = {:.6f}".
                  format(epoch_num+1, keras.backend.get_value(self.model.optimizer.lr)))

        # The learning rate values may need to be adjusted when you configure different number of epochs,
        # change between Full or Tiny model, or significant changes of training data size. The goal of
        # using bigger learning rate at the first a few epochs is not only to speed up the training, but
        # mainly to lower the chance of falling into local optima.
        if epoch_num < 1:
            return 4e-4
        elif epoch_num < 2:
            return 3.2e-4
        elif epoch_num < 4:
            return 2.4e-4
        elif epoch_num < 6:
            return 2e-4
        elif epoch_num < 10:
            return 1.6e-4
        elif epoch_num < 16:
            return 1.2e-4
        elif epoch_num < 24:
            return 1.1e-4
        elif epoch_num < 40:
            return 1e-4
        else:
            return 9.6e-5

    def _decode_netout(self, netout):
        grid_h, grid_w = netout.shape[:2]

        conf_boxes = {}
        for row in range(grid_h):
            for col in range(grid_w):
                # first 4 elements are x, y, w, and h
                x, y, w, h = netout[row, col, :4]

                x = (col + self.sigmoid(x)) * self.config['grid_x_size'] + self.config['image_left_skip']
                y = (row + self.sigmoid(y)) * self.config['grid_y_size']
                w = w * self.config['grid_x_size']
                h = h * self.config['grid_y_size']

                confidence = self.sigmoid(netout[row, col, 4])

                if self.config['debug'] and confidence > 0.1:
                    print("Net out: {}, {}, {}, {}, {}".format(x, y, w, h, confidence))

                if confidence > 0.5:
                    box = BoundBox(x, y, w, h)
                    # Check if this new box is obviously (20%) overlapping with any existing boxes in
                    # the list. If so, keep the one with higher confidence. Note that as the boxes reach
                    # this confidence are very few, therefore, the looping efficiency is not a concern.
                    # Also, in our use case, there are not supposed to have any overlapping boxes in the
                    # prediction. This should also apply for most typical OCR applications.
                    redundant = False
                    for bx, conf in conf_boxes.items():
                        if box.get_box_iou_with(bx) > 0.2:
                            if conf < confidence:
                                conf_boxes[bx] = 0.0  # Will be discarded later
                                # Continue to check if this new one is overlapping with any others
                            else:
                                redundant = True
                                break
                    if not redundant:
                        conf_boxes[box] = confidence

        boxes = []
        for bx, conf in conf_boxes.items():
            if conf > 0.5:
                boxes.append(bx)

        return boxes

    @staticmethod
    def custom_loss(y_true, y_pred):
        # Get prediction
        pred_box_xy = tf.sigmoid(y_pred[..., :2])
        pred_box_wh = y_pred[..., 2:4]
        pred_box_conf = tf.sigmoid(y_pred[..., 4])

        # Get ground truth
        true_box_xy = y_true[..., :2]
        true_box_wh = y_true[..., 2:4]
        true_box_conf = y_true[..., 4]

        # Determine the mask: simply the position of the ground truth boxes (the predictors)
        true_mask = tf.expand_dims(y_true[..., 4], axis=-1)

        # Calculate the loss. A scale can be associated with each loss, indicating how important
        # the loss is. The bigger the scale, more important the loss is.
        loss_xy = tf.reduce_sum(tf.square(true_box_xy - pred_box_xy) * true_mask) * 1.0
        loss_wh = tf.reduce_sum(tf.square(true_box_wh - pred_box_wh) * true_mask) * 1.0
        loss_conf = tf.reduce_sum(tf.square(true_box_conf - pred_box_conf)) * 1.2

        loss = loss_xy + loss_wh + loss_conf
        return loss

    @staticmethod
    def normalize(image):
        return image / 255.

    @staticmethod
    def sigmoid(x):
        return 1. / (1. + np.exp(-x))


class BoundBox:
    def __init__(self, cx, cy, w, h):
        self.cx = cx
        self.cy = cy
        self.w = w
        self.h = h

    def get_coordinates(self):
        xmin = math.floor(self.cx - self.w/2.)
        ymin = math.floor(self.cy - self.h/2.)
        xmax = math.ceil(self.cx + self.w/2.)
        ymax = math.ceil(self.cy + self.h/2.)

        return xmin, ymin, xmax, ymax

    def get_box_iou_with(self, box2):
        x1_min, y1_min, x1_max, y1_max = self.get_coordinates()
        x2_min, y2_min, x2_max, y2_max = box2.get_coordinates()

        intersect_w = self._interval_overlap([x1_min, x1_max], [x2_min, x2_max])
        intersect_h = self._interval_overlap([y1_min, y1_max], [y2_min, y2_max])

        intersect = intersect_w * intersect_h

        union = self.w * self.h + box2.w * box2.h - intersect

        return float(intersect) / union

    @staticmethod
    def _interval_overlap(interval_a, interval_b):
        x1, x2 = interval_a
        x3, x4 = interval_b

        if x3 < x1:
            if x4 < x1:
                return 0
            else:
                return min(x2, x4) - x1
        else:
            if x2 < x3:
                return 0
            else:
                return min(x2, x4) - x3
