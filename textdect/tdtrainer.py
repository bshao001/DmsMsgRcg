import numpy as np
import os

from settings import PROJECT_ROOT
from misc.imgreader import ImgReader
from misc.imgconvnets import ImgConvNets

FEATURE_HEIGHT = 28
FEATURE_WIDTH = 28


def train(learning_rate, lr_adaptive, max_steps, result_file, last_file=None, retrain=False):
    height, width = FEATURE_HEIGHT, FEATURE_WIDTH

    pos_feats, neg_feats = read_features_from_files(height, width)

    pos_y = np.ones((pos_feats.shape[0], 1), dtype=np.float32)
    neg_y = np.zeros((neg_feats.shape[0], 1), dtype=np.float32)

    all_feats = np.append(pos_feats, neg_feats, axis=0)
    all_y = np.append(pos_y, neg_y, axis=0)

    print("all_feats shapes: pos = {}, neg = {}, all = {}; and dtype = {}".format(
        pos_feats.shape, neg_feats.shape, all_feats.shape, all_feats.dtype))
    print("all_y shape: {}; and dtype={}".format(all_y.shape, all_y.dtype))

    res_dir = os.path.join(PROJECT_ROOT, 'Data', 'Result')
    img_cnn = ImgConvNets(2, keep_prob=0.5, batch_size=64, learning_rate=learning_rate,
                          lr_adaptive=lr_adaptive, max_steps=max_steps, model='STCNN')
    if retrain:
        img_cnn.retrain(all_feats, all_y, height, width, res_dir,
                        last_file=last_file, new_file=result_file)
    else:
        img_cnn.train(all_feats, all_y, height, width, res_dir,
                      result_file=result_file)


def read_features_from_files(height, width, folder='Training'):
    img_reader = ImgReader(height, width)

    pos_dir = os.path.join(PROJECT_ROOT, 'Data', 'Step1', folder, 'positive')
    pos_feats = np.asarray(
        img_reader.get_features_all_images(pos_dir, skip=[0, 0, 0, 0], stride=1, data_augm=False),
        dtype=np.float32)

    if folder=='Training':
        neg1_dir = os.path.join(PROJECT_ROOT, 'Data', 'Step1', folder, 'negative1')
        neg5_dir = os.path.join(PROJECT_ROOT, 'Data', 'Step1', folder, 'negative5')
        neg1_feats = img_reader.get_features_all_images(neg1_dir, skip=[0, 0, 0, 0], stride=1,
                                                        padding=False)
        neg5_feats = img_reader.get_features_all_images(neg5_dir, skip=[0, 0, 0, 0], stride=5,
                                                        padding=False)
        neg_feats = np.asarray(np.append(neg1_feats, neg5_feats, axis=0), dtype=np.float32)
    else:
        neg_dir = os.path.join(PROJECT_ROOT, 'Data', 'Step1', folder, 'negative')
        neg_feats = np.asarray(
            img_reader.get_features_all_images(neg_dir, skip=[0, 0, 0, 0], stride=5, padding=False),
            dtype=np.float32)

    return pos_feats, neg_feats


if __name__ == "__main__":
    from time import time
    import tensorflow as tf
    from misc.cnnpredictor import *

    training = False
    if training:
        t0 = time()
        train(learning_rate=1e-4, lr_adaptive=True, max_steps=330000, result_file='step1_dcnn')
        t1 = time()
        print("Training time: {:6.2f} seconds".format(t1 - t0))
    else:
        height, width = FEATURE_HEIGHT, FEATURE_WIDTH
        pos_feats, neg_feats = read_features_from_files(height, width, folder='Validation')

        pos_cnt = pos_feats.shape[0]
        neg_cnt = neg_feats.shape[0]

        res_dir = os.path.join(PROJECT_ROOT, 'Data', 'Result')
        model_list = get_all_models(res_dir, "step1_")

        with tf.Session() as sess:
            for model in model_list:
                cnn_pred = CnnPredictor(sess, res_dir, model)
                _, pos_arr = cnn_pred.predict(pos_feats)
                _, neg_arr = cnn_pred.predict(neg_feats)

                pos_err = 0
                for i in range(pos_cnt):
                    if pos_arr[i][0] == 0:
                        pos_err += 1

                neg_err = 0
                for i in range(neg_cnt):
                    if neg_arr[i][0] == 1:
                        neg_err += 1

                all_err = pos_err + neg_err
                all_cnt = pos_cnt + neg_cnt

                print("Result of model: {}".format(model))
                print("Errors made on positive samples: {} out of {}, accuracy = {:5.2f}%"
                      .format(pos_err, pos_cnt, ((pos_cnt-pos_err)/pos_cnt)*100))
                print("Errors made on negative samples: {} out of {}, accuracy = {:5.2f}%"
                      .format(neg_err, neg_cnt, ((neg_cnt-neg_err)/neg_cnt)*100))
                print("Errors made on all samples: {} out of {}, overall accuracy = {:5.2f}%"
                      .format(all_err, all_cnt, ((all_cnt-all_err)/all_cnt)*100))
                print()