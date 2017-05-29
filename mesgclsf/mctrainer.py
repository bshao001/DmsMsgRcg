import numpy as np
import os

from settings import PROJECT_ROOT
from misc.imgreader import ImgReader
from misc.imgconvnets import ImgConvNets

FEATURE_HEIGHT = 28
FEATURE_WIDTH = 32


def train_tas(model, learning_rate, lr_adaptive, max_steps, result_file, last_file=None,
              retrain=False):
    height, width = FEATURE_HEIGHT, FEATURE_WIDTH

    feats0, feats1 = read_features_tas(height, width)

    y0 = np.zeros((feats0.shape[0], 1), dtype=np.float32)
    y1 = np.ones((feats1.shape[0], 1), dtype=np.float32)

    all_feats = np.append(feats0, feats1, axis=0)
    all_y = np.append(y0, y1, axis=0)

    print("all_feats shapes: toll = {}, closed = {}, all = {}; "
          "and dtype = {}".format(feats0.shape, feats1.shape, all_feats.shape, all_feats.dtype))
    print("all_y shape: {}; and dtype={}".format(all_y.shape, all_y.dtype))

    res_dir = os.path.join(PROJECT_ROOT, 'Data', 'Result')
    img_cnn = ImgConvNets(model=model, class_count=2, keep_prob=0.5, batch_size=32,
                          learning_rate=learning_rate, lr_adaptive=lr_adaptive,
                          max_steps=max_steps)
    if retrain:
        img_cnn.retrain(all_feats, all_y, height, width, res_dir,
                        last_file=last_file, new_file=result_file)
    else:
        img_cnn.train(all_feats, all_y, height, width, res_dir,
                      result_file=result_file)


def train_lss(model, learning_rate, lr_adaptive, max_steps, result_file, last_file=None,
              retrain=False):
    height, width = FEATURE_HEIGHT, FEATURE_WIDTH

    feats0, feats1, feats2, feats3 = read_features_lss(height, width)

    y0 = np.zeros((feats0.shape[0], 1), dtype=np.float32)
    y1 = np.ones((feats1.shape[0], 1), dtype=np.float32)
    y2 = np.ones((feats2.shape[0], 1), dtype=np.float32) * 2
    y3 = np.ones((feats3.shape[0], 1), dtype=np.float32) * 3

    all_feats = np.append(np.append(np.append(feats0, feats1, axis=0), feats2, axis=0),
                          feats3, axis=0)
    all_y = np.append(np.append(np.append(y0, y1, axis=0), y2, axis=0), y3, axis=0)

    print("all_feats shapes: zero toll = {}, closed = {}, normal = {}, congested = {},  all = {}; "
          "and dtype = {}".format(feats0.shape, feats1.shape, feats2.shape, feats3.shape,
                                  all_feats.shape, all_feats.dtype))
    print("all_y shape: {}; and dtype={}".format(all_y.shape, all_y.dtype))

    res_dir = os.path.join(PROJECT_ROOT, 'Data', 'Result')
    img_cnn = ImgConvNets(model=model, class_count=4, keep_prob=0.5, batch_size=32,
                          learning_rate=learning_rate, lr_adaptive=lr_adaptive,
                          max_steps=max_steps, )
    if retrain:
        img_cnn.retrain(all_feats, all_y, height, width, res_dir,
                        last_file=last_file, new_file=result_file)
    else:
        img_cnn.train(all_feats, all_y, height, width, res_dir,
                      result_file=result_file)


def read_features_tas(height, width, folder='Training'):
    base_dir = os.path.join(PROJECT_ROOT, 'Data', 'Step2', folder, 'TasMsg')

    toll_dir =  os.path.join(base_dir, 'Toll0')
    closed_dir = os.path.join(base_dir, 'Closed1')

    img_reader = ImgReader(height, width)
    toll_feats = np.asarray(
        img_reader.get_features_all_images(toll_dir, skip=[0, 0, 0, 0], stride=2),
        dtype=np.float32)
    closed_feats = np.asarray(
        img_reader.get_features_all_images(closed_dir, skip=[0, 0, 0, 0], stride=1),
        dtype=np.float32)

    return toll_feats, closed_feats


def read_features_lss(height, width, folder='Training'):
    base_dir = os.path.join(PROJECT_ROOT, 'Data', 'Step2', folder, 'LssMsg')

    zerotoll_dir =  os.path.join(base_dir, 'ZeroToll0')
    closed_dir = os.path.join(base_dir, 'Closed1')
    normal_dir = os.path.join(base_dir, 'Normal2')
    congested_dir = os.path.join(base_dir, 'Congested3')

    img_reader = ImgReader(height, width)
    zerotoll_feats = np.asarray(
        img_reader.get_features_all_images(zerotoll_dir, skip=[0, 0, 0, 0], stride=1),
        dtype=np.float32)
    closed_feats = np.asarray(
        img_reader.get_features_all_images(closed_dir, skip=[0, 0, 0, 0], stride=1),
        dtype=np.float32)
    normal_feats = np.asarray(
        img_reader.get_features_all_images(normal_dir, skip=[0, 0, 0, 0], stride=1),
        dtype=np.float32)
    congested_feats = np.asarray(
        img_reader.get_features_all_images(congested_dir, skip=[0, 0, 0, 0], stride=1),
        dtype=np.float32)

    return zerotoll_feats, closed_feats, normal_feats, congested_feats

if __name__ == "__main__":
    from time import time
    import tensorflow as tf
    from misc.cnnpredictor import CnnPredictor

    training = True
    data_type = 'TAS'

    if training:
        t0 = time()
        if data_type == 'TAS':
            train_tas(model='BASIC', learning_rate=1e-4, lr_adaptive=True, max_steps=8000,
                      result_file='step2_tas_basic', retrain=False)
        else:
            train_lss(model='BASIC', learning_rate=1e-4, lr_adaptive=True, max_steps=8000,
                      result_file='step2_lss_basic', retrain=False)

        t1 = time()
        print("Training time: {:6.2f} seconds".format(t1 - t0))
    else:
        height, width = FEATURE_HEIGHT, FEATURE_WIDTH
        res_dir = os.path.join(PROJECT_ROOT, 'Data', 'Result')

        if data_type == 'TAS':
            f0, f1 = read_features_tas(height, width, folder='Validation')
            cnt0, cnt1 = f0.shape[0], f1.shape[0]

            with tf.Session() as sess:
                cnn_pred = CnnPredictor(sess, res_dir, 'step2_tas_basic')
                _, ind0 = cnn_pred.predict(f0)
                _, ind1 = cnn_pred.predict(f1)

            err0 = 0
            for i in range(cnt0):
                if ind0[i][0] != 0:
                    err0 += 1

            err1 = 0
            for i in range(cnt1):
                if ind1[i][0] != 1:
                    err1 += 1

            all_err = err0 + err1
            all_cnt = cnt0 + cnt1
            print("Errors made on toll($) messages: {} out of {}, accuracy = {:5.2f}%"
                  .format(err0, cnt0, ((cnt0 - err0) / cnt0) * 100))
            print("Errors made on closed messages: {} out of {}, accuracy = {:5.2f}%"
                  .format(err1, cnt1, ((cnt1 - err1) / cnt1) * 100))
            print("Errors made on all messages: {} out of {}, accuracy = {:5.2f}%"
                  .format(all_err, all_cnt, ((all_cnt - all_err) / all_cnt) * 100))
        else:
            f0, f1, f2, f3 = read_features_lss(height, width, folder='Training')
            cnt0, cnt1, cnt2, cnt3 = f0.shape[0], f1.shape[0], f2.shape[0], f3.shape[0]

            with tf.Session() as sess:
                cnn_pred = CnnPredictor(sess, res_dir, 'step2_lss_basic')
                _, ind0 = cnn_pred.predict(f0)
                _, ind1 = cnn_pred.predict(f1)
                _, ind2 = cnn_pred.predict(f2)
                _, ind3 = cnn_pred.predict(f3)

            err0 = 0
            for i in range(cnt0):
                 if ind0[i][0] != 0:
                    err0 += 1

            err1 = 0
            for i in range(cnt1):
                if ind1[i][0] != 1:
                    err1 += 1

            err2 = 0
            for i in range(cnt2):
                if ind2[i][0] != 2:
                    err2 += 1

            err3 = 0
            for i in range(cnt3):
                if ind3[i][0] != 3:
                    err3 += 1

            all_err = err0 + err1 + err2 + err3
            all_cnt = cnt0 + cnt1 + cnt2 + cnt3
            print("Errors made on zero toll messages: {} out of {}, accuracy = {:5.2f}%"
                  .format(err0, cnt0, ((cnt0 - err0) / cnt0) * 100))
            print("Errors made on closed messages: {} out of {}, accuracy = {:5.2f}%"
                  .format(err1, cnt1, ((cnt1 - err1) / cnt1) * 100))
            print("Errors made on normal messages: {} out of {}, accuracy = {:5.2f}%"
                  .format(err2, cnt2, ((cnt2 - err2) / cnt2) * 100))
            print("Errors made on congested messages: {} out of {}, accuracy = {:5.2f}%"
                  .format(err3, cnt3, ((cnt3 - err3) / cnt3) * 100))
            print("Errors made on all messages: {} out of {}, accuracy = {:5.2f}%"
                  .format(all_err, all_cnt, ((all_cnt - all_err) / all_cnt) * 100))