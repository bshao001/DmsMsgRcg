import numpy as np
import tensorflow as tf
from collections import Counter

from textdect.convertmodel import ConvertedModel
from misc.freezemodel import FreezedModel
from mesgclsf.datapreptools import resize_to_desired
from mesgclsf.s2train import FEATURE_HEIGHT, FEATURE_WIDTH


def classify(classifier, session, img_arr, stride=16):
    """
    Take an image file (an output from the first step), read and analyze the image, and 
    returns the class ID of the input image based on the message content on it.
    Args:
        classifier: A FreezedModel constructed based on a trained model for step 2.
        session: The TensorFlow session.
        img_arr: A numpy ndarray that holds the image features.
        stride: Optional. The stride of the sliding.
    Returns:
        class_id: Message class ID of the input image.
        confidence: A percentage indicating how many slided images were predicted as
            this class identified by the class_id.
    """
    height, width = FEATURE_HEIGHT, FEATURE_WIDTH

    resized_img = resize_to_desired(img_arr)
    img_height, img_width = resized_img.shape
    assert img_height == height

    features = []
    for x in range(0, img_width - FEATURE_WIDTH + 1, stride):
        this_win = resized_img[:, x:x + FEATURE_WIDTH]
        features.append(this_win.reshape(-1))

    _, indices = classifier.predict(session, np.asarray(features))

    img_cnt = len(features)

    cls_list = []
    for i in range(img_cnt):
        cls_list.append(indices[i][0])

    class_id, pos_cnt = Counter(cls_list).most_common()[0]
    confidence = (pos_cnt / img_cnt) * 100.0

    return class_id, confidence


def detect_and_classify(detector, classifier, session, image_array, debug=False):
    boxes = detector.predict(session, image_array)
    for box in boxes:
        xmin, ymin, xmax, ymax = box.get_coordinates()
        gry_arr = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

        area_img = gry_arr[ymin:ymax, xmin:xmax]
        cls_id, conf = classify(classifier, session, area_img)

        if debug:
            cv2.rectangle(image_array, (xmin, ymin), (xmax, ymax), (255, 0, 0), 1)
            print("class ID: {}, confidence: {}".format(cls_id, conf))


if __name__ == "__main__":
    import cv2
    import json
    import os
    import matplotlib.pyplot as plt

    from time import time

    from settings import PROJECT_ROOT

    t0 = time()

    model_dir = os.path.join(PROJECT_ROOT, 'Data', 'Result')
    s1_model = 's1_keras_model.pb'
    lss_model = 's2_lss_model.pb'
    tas_model = 's2_tas_model.pb'

    sign_type = 'LSS'  # or TAS
    img_file = os.path.join(PROJECT_ROOT, 'Data', 'Step1', 'Test', '10.jpg')
    img_arr = cv2.imread(img_file)

    s1_config_file = os.path.join(PROJECT_ROOT, 'textdect', 'config.json')
    with open(s1_config_file) as config_buffer:
        s1_config = json.loads(config_buffer.read())

    with tf.Graph().as_default() as graph:
        detector = ConvertedModel(s1_config, graph, 's1_keras', model_dir, s1_model)
        lss_classifier = FreezedModel(graph, 's2_lss', model_dir, lss_model)
        tas_classifier = FreezedModel(graph, 's2_tas', model_dir, tas_model)

    with tf.Session(graph=graph) as sess:
        if sign_type == 'LSS':
            detect_and_classify(detector, lss_classifier, sess, img_arr, debug=True)
        else:
            detect_and_classify(detector, tas_classifier, sess, img_arr, debug=True)

    t1 = time()
    print("Running time: {:4.2f} seconds".format(t1-t0))

    plt.imshow(img_arr)
    plt.show()

