import numpy as np
import tensorflow as tf
from collections import Counter

from misc.cnnpredictor import CnnPredictor
from mesgclsf.imageresizer import resize_to_desired
from mesgclsf.mctrainer import FEATURE_HEIGHT, FEATURE_WIDTH


def classify(session, result_dir, result_file, img_arr, stride=16):
    """
    Take an image file, read and analyze the image, and returns the class ID of the
    input image based on the message content on it.
    Args:
        session: The TensorFlow session used to run the prediction.
        result_dir: The full path to the folder in which the result file locates.
        result_file: The file that saves the training results.
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
        this_win = img_arr[:, x:x + FEATURE_WIDTH]
        features.append(this_win.reshape(-1))

    cnn_pred = CnnPredictor(session, result_dir, result_file)
    _, indices = cnn_pred.predict(np.asarray(features))

    img_cnt = len(features)

    cls_list = []
    for i in range(img_cnt):
        cls_list.append(indices[i][0])

    class_id, pos_cnt = Counter(cls_list).most_common()[0]
    confidence = (pos_cnt / img_cnt) * 100.0

    return class_id, confidence


if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt
    from skimage import io as skio
    from skimage import color as skcolor
    from time import time

    from settings import PROJECT_ROOT
    from misc.imgtools import plot_samples

    t0 = time()
    res_dir = os.path.join(PROJECT_ROOT, 'Data', 'Result')
    img_file_name = os.path.join(PROJECT_ROOT, 'Data', 'Step1', 'Test', 'sign14.jpg')
    # with tf.Session() as sess:
    #     areas, pos_a = detect(sess, res_dir, 'step1_deep_v2', img_file_name)
    t1 = time()
    print("Running time: {:4.2f} seconds".format(t1-t0))

