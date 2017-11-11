import numpy as np
import tensorflow as tf
from skimage import filters as flt
from time import time

from misc.cnnpredictor import CnnPredictor
from misc.imgreader import ImgReader
from textdect.tdtrainer import FEATURE_HEIGHT, FEATURE_WIDTH


def detect(detector, gray_array, skip=[72, 100, 72, 116], stride=5, sample_limit=5, trim=True,
           limit_ratio=1.5, padding=4, debug=False):
    """
    Take an image file, read and analyze the image, and returns the pixel indexes
    of all areas that contain DMS message texts.
    Args:
        detector: A CnnPredictor constructed based on a trained model for step 1.
        gray_array: The input image (as a numpy ndarray) to be analyzed, in gray color.
        skip: Optional. The pixels to be ignored in the border areas in each side, assuming
            no useful information is contained in these areas. Fixed to be 4 numbers, in
            the order of top, right, bottom, and left.
        stride: Optional. The stride of the sliding.
        sample_limit: Optional. If sample_limit number of slided images are not predicted
            as positive samples, this line will be ignored. Otherwise, it will contribute
            to expand the text area.
        trim: Optional. Whether to trim the output areas (only horizontally).
        limit_ratio: Optional. Unless padded, all pixels sitting in the left or right sides
            of an output area with STD values lower than the maximum STD values in the whole
            area will be trimmed.
        padding: Optional. Keep this amount of pixels (otherwise be trimmed) in the output.
        debug: Optional. Whether to output debug information.
    Returns:
        A list of lists, each of which is a 4-numbers list describing a rectangle
        area indicating message texts contained inside: top_y, top_x, height, and
        width. top_y and top_x are offsets based on the original image (not the one
        that may have pixels skipped).
    """
    height, width = FEATURE_HEIGHT, FEATURE_WIDTH

    img_reader = ImgReader(height, width)
    coords, feats = img_reader.get_image_array_features(gray_array, skip, stride, padding=False)

    feat_images = np.asarray(feats)
    img_cnt = feat_images.shape[0]
    if debug:
        print("img_cnt = {}".format(img_cnt))

    t0 = time()
    _, indices = detector.predict(feat_images)
    t1 = time()
    if debug:
        print("Prediction time: {}".format(t1-t0))

    pos_areas = []  # Output for debugging purpose
    hor_lines = []  # Element in format of y, x1, x2, sample_cnt
    for i in range(img_cnt):
        if indices[i][0] == 1:
            y, x = coords[i][0], coords[i][1]
            pos_areas.append([y, x])

            added = False
            for h in range(len(hor_lines)):
                hy, hx1, hx2, cnt = hor_lines[h][0], hor_lines[h][1], hor_lines[h][2], hor_lines[h][3]
                if y == hy and hx1 < x < hx2 < x+width:
                    hor_lines[h] = [hy, hx1, x+width, cnt+1]
                    added = True
                    break

            if not added:
                hor_lines.append([y, x, x+width, 1])

    if debug:
        for h in hor_lines:
            print("Line: {}, {}, {}, {}".format(h[0], h[1], h[2], h[3]))

    text_areas = []
    pre_y, pre_x, pre_h, pre_w = 0, 0, 0, 0
    for i in range(len(hor_lines)):
        y, x, x2, cnt = hor_lines[i][0], hor_lines[i][1], hor_lines[i][2], hor_lines[i][3]
        if cnt < sample_limit:
            continue

        if pre_y == 0: # First valid line
            pre_y, pre_x, pre_h, pre_w = y, x, height, x2 - x
        elif (y + height) - (pre_y + pre_h) <= 2*stride:  # Still connected to the last line
            # No touch to pre_y, and update all others
            pre_x = min(x, pre_x)
            pre_h += stride
            pre_w = max(x2, pre_x + pre_w) - pre_x
        else: # A line of a new text area
            text_areas.append([pre_y, pre_x, pre_h, pre_w])
            pre_y, pre_x, pre_h, pre_w = y, x, height, x2 - x

    if pre_y > 0:
        text_areas.append([pre_y, pre_x, pre_h, pre_w])

    if trim:
        new_areas = []
        for area in text_areas:
            new_areas.append(_std_trim(gray_array, area, limit_ratio, padding))

        return new_areas, pos_areas
    else:
        return text_areas, pos_areas


def _std_trim(img_arr, text_area, limit_ratio=1.5, padding=4):
    y, x, h, w = text_area[0], text_area[1], text_area[2], text_area[3]
    x2, y2 = x + w, y + h

    cropped_img = img_arr[y:y2, x:x2]
    filtered_img = np.array(cropped_img > flt.threshold_li(cropped_img))

    # Trim the filtered_img vertically
    ver_list = []
    for v in range(0, y2-y):
        std = np.std(filtered_img[v, :])
        ver_list.append(std)

    limit_ver = max(ver_list) / limit_ratio
    new_y, new_y2 = y, y2
    ver_size = len(ver_list)

    for i in range(ver_size):
        if ver_list[i] > limit_ver or new_y2 - new_y <= FEATURE_HEIGHT:
            break
        if i >= padding: new_y += 1

    for j in range(ver_size-1, 0, -1):
        if ver_list[j] > limit_ver or new_y2 - new_y <= FEATURE_HEIGHT:
            break
        if j <= ver_size-1-padding: new_y2 -= 1

    ver_out_img = filtered_img[new_y-y:new_y2-y, :]

    # Trim the ver_out_img horizontally
    hor_list = []
    for h in range(0, x2-x):
        std = np.std(ver_out_img[:, h])
        hor_list.append(std)

    limit_hor = max(hor_list) / limit_ratio
    new_x, new_x2 = x, x2
    hor_size = len(hor_list)

    for i in range(hor_size):
        if hor_list[i] > limit_hor or new_x2 - new_x <= FEATURE_WIDTH:
            break
        if i >= padding: new_x += 1

    for j in range(hor_size-1, 0, -1):
        if hor_list[j] > limit_hor or new_x2 - new_x <= FEATURE_WIDTH:
            break
        if j <= hor_size-1-padding: new_x2 -= 1

    return [new_y, new_x, new_y2-new_y, new_x2-new_x]


if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt
    from skimage import io as skio
    from skimage import color as skcolor

    from settings import PROJECT_ROOT
    from misc.imgtools import plot_samples

    t0 = time()

    res_dir = os.path.join(PROJECT_ROOT, 'Data', 'Result')
    img_file_name = os.path.join(PROJECT_ROOT, 'Data', 'Temp', 'OtmImages', 's_193445_2.jpg')
    img_arr = skio.imread(img_file_name)
    gray_arr = skcolor.rgb2gray(img_arr)

    with tf.Session() as sess:
        detector = CnnPredictor(sess, 's1', res_dir, 'step1_basic_s1')
        areas, pos_a = detect(detector, gray_arr)

    t1 = time()
    print("Running time: {:4.2f} seconds".format(t1-t0))

    debug = False
    if debug:
        pos_feats = []
        for p in pos_a:
            y, x = p[0], p[1]
            print("Pos area: {}, {}".format(y, x))
            feat_row = gray_arr[y:y+FEATURE_HEIGHT, x:x+FEATURE_WIDTH].reshape(-1)
            pos_feats.append(feat_row)

        X_images = np.asarray(pos_feats)
        plot_samples(X_images, img_height=FEATURE_HEIGHT, img_width=FEATURE_WIDTH,
                     figsize=(3.2, 3.2), transpose=False, shuffle=False)
    else:
        for area in areas:
            y, x, h, w = area[0], area[1], area[2], area[3]
            print("Area (y, x, h, w): {}, {}, {}, {}".format(y, x, h, w))
            x2, y2 = x+w, y+h

            img_arr[y:y+2, x:x2] = [255, 0, 0]  # Top line
            img_arr[y:y2, x2:x2+2] = [255, 0, 0]  # Right Line
            img_arr[y2:y2 + 2, x:x2] = [255, 0, 0]  # Bottom Line
            img_arr[y:y2, x:x+2] = [255, 0, 0]  # Left line

        plt.imshow(img_arr)
        plt.show()