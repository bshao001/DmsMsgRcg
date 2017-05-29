import numpy as np
import tensorflow as tf
from collections import Counter

from misc.cnnpredictor import CnnPredictor
from textdect.textdetector import detect
from mesgclsf.imageresizer import resize_to_desired
from mesgclsf.mctrainer import FEATURE_HEIGHT, FEATURE_WIDTH


def classify(classifier, img_arr, stride=16):
    """
    Take an image file (an output from the first step), read and analyze the image, and 
    returns the class ID of the input image based on the message content on it.
    Args:
        classifier: A CnnPredictor constructed based on a trained model for step 2.
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

    _, indices = classifier.predict(np.asarray(features))

    img_cnt = len(features)

    cls_list = []
    for i in range(img_cnt):
        cls_list.append(indices[i][0])

    class_id, pos_cnt = Counter(cls_list).most_common()[0]
    confidence = (pos_cnt / img_cnt) * 100.0

    return class_id, confidence


def detect_and_classify(result_dir, detect_file, classify_file, gray_array):
    with tf.Session() as s1:
        detector = CnnPredictor(s1, result_dir, detect_file)
        areas, _ = detect(detector, gray_array)

    tf.reset_default_graph()
    with tf.Session() as s2:
        classifier = CnnPredictor(s2, result_dir, classify_file)
        for area in areas:
            y, x, h, w = area[0], area[1], area[2], area[3]
            x2, y2 = x + w, y + h

            area_img = gry_arr[y:y2, x:x2]
            cls_id, conf = classify(classifier, area_img)
            print("class ID: {}, confidence: {}".format(cls_id, conf))

if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt
    from skimage import io as skio
    from skimage import color as skcolor
    from time import time

    from settings import PROJECT_ROOT

    t0 = time()
    res_dir = os.path.join(PROJECT_ROOT, 'Data', 'Result')
    img_file = os.path.join(PROJECT_ROOT, 'Data', 'Step1', 'Test', 'sign1.jpg')
    img_arr = skio.imread(img_file)
    gry_arr = skcolor.rgb2gray(img_arr)

    detect_and_classify(sess, res_dir, 'step1_dcnn', 'step2_tas_basic', gry_arr)

    t1 = time()
    print("Running time: {:4.2f} seconds".format(t1-t0))

    plt.imshow(img_arr)
    plt.show()

