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
If you convert a model that was not trained on the same computer, make sure the python versions
on the two machines are the same (all in 3.5 or all in 3.6). Otherwise, you may encounter weird
errors.
"""
import cv2
import json
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras

from textdect.yolonet import YoloNet
from textdect.s1predict import draw_boxes


class ConvertedModel(object):
    def __init__(self, config, graph, model_scope, model_dir, model_file):
        self.config = config

        frozen_model = os.path.join(model_dir, model_file)
        with tf.gfile.GFile(frozen_model, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        # This model_scope adds a prefix to all the nodes in the graph
        tf.import_graph_def(graph_def, input_map=None, return_elements=None,
                            name="{}/".format(model_scope))

        # Uncomment the two lines below to look for the names of all the operations in the graph
        # for op in graph.get_operations():
        #    print(op.name)

        # Using the lines commented above to look for the tensor name of the input node
        # Or you can figure it out in your original model, if you explicitly named it.
        self.input_tensor = graph.get_tensor_by_name("{}/input_1:0".format(model_scope))
        self.output_tensor = graph.get_tensor_by_name("{}/s1_output0:0".format(model_scope))

    def predict(self, session, image):
        input_image = YoloNet.normalize(image[:, self.config['image_left_skip']:-self.config['image_right_skip'],
                                        ::-1])
        input_image = np.expand_dims(input_image, 0)
        # This session.run line corresponds to model.predict() method in Keras. For most other
        # models, you only need this line to work. All others are specific to this application.
        netout = session.run(self.output_tensor,
                             feed_dict={self.input_tensor: input_image})[0]
        boxes = YoloNet.decode_netout(self.config, netout)

        return boxes


def s1_predict(config_file, model_dir, model_file, predict_file_list, out_dir):
    """
    This function serves as a test/validation tool during the model development. It is not used as
    a final product in part of the pipeline.
    """
    with open(config_file) as config_buffer:
        config = json.loads(config_buffer.read())

    with tf.Graph().as_default() as graph:
        converted_model = ConvertedModel(config, graph, 's1_keras', model_dir, model_file)

    with tf.Session(graph=graph) as sess:
        for img_file in predict_file_list:
            image = cv2.imread(img_file)
            boxes = converted_model.predict(sess, image)
            image = draw_boxes(image, boxes)

            _, filename = os.path.split(img_file)
            cv2.imwrite(os.path.join(out_dir, filename), image)


def convert(model_dir, keras_model_file, tf_model_file, name_output='s1_output', num_output=1):
    # Parameter False is for tf.keras in TF 1.4. For real Keras, use 0 as parameter
    keras.backend.set_learning_phase(False)
    keras_model = keras.models.load_model(os.path.join(model_dir, keras_model_file),
                                          custom_objects={'custom_loss': YoloNet.custom_loss})

    output = [None] * num_output
    out_node_names = [None] * num_output
    for i in range(num_output):
        out_node_names[i] = name_output + str(i)
        output[i] = tf.identity(keras_model.outputs[i], name=out_node_names[i])

    sess = keras.backend.get_session()
    constant_graph = tf.graph_util.convert_variables_to_constants(
        sess,
        sess.graph.as_graph_def(),
        out_node_names  # All other operations relying on this will also be saved
    )
    output_file = os.path.join(model_dir, tf_model_file)
    with tf.gfile.GFile(output_file, "wb") as f:
        f.write(constant_graph.SerializeToString())

    print("Converted model was saved as {}.".format(tf_model_file))


if __name__ == '__main__':
    from settings import PROJECT_ROOT

    action = 'predict'  # Modify this line to run convert or predict
    if action == 'convert':
        model_dir = os.path.join(PROJECT_ROOT, 'Data', 'Result')
        keras_model = 's1_model_weights.h5'  # model architecture and weights
        # will use model_scope as the first part of the filename of the .pb files across the whole project
        tf_model = 's1_keras_model.pb'
        convert(model_dir, keras_model, tf_model)
    else:  # predict
        model_dir = os.path.join(PROJECT_ROOT, 'Data', 'Result')
        model_file = 's1_keras_model.pb'
        img_dir = os.path.join(PROJECT_ROOT, 'Data', 'OtmImages')
        out_dir = os.path.join(PROJECT_ROOT, 'Data', 'Temp')

        file_list = []
        file_count = 0
        for img_file in sorted(os.listdir(img_dir)):
            full_path_name = os.path.join(img_dir, img_file)
            if os.path.isfile(full_path_name) and img_file.lower().endswith(tuple(['.jpg', '.png'])):
                file_count += 1
                if file_count > 200:
                    file_list.append(full_path_name)
                    if file_count >= 1000:
                        break

        s1_predict('config.json', model_dir, model_file, file_list, out_dir)

