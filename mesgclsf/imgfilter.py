import math
import matplotlib.pyplot as plt
import numpy as np
import os

from skimage import color as skcolor
from skimage import exposure as exps
from skimage import filters as flts
from skimage.morphology import disk
from skimage.filters.rank import enhance_contrast
from skimage import io as skio

from settings import PROJECT_ROOT
from mesgclsf.imageresizer import get_immediate_subfolders

if __name__ == "__main__":
    ext_filter = ['.jpg', '.png']
    in_dir = os.path.join(PROJECT_ROOT, 'Data', 'Step2', 'Validation')

    res_imgs = []
    for sub_dir in get_immediate_subfolders(in_dir):
        full_dir = os.path.join(in_dir, sub_dir)
        for sub_dir2 in get_immediate_subfolders(full_dir):
            full_dir2 = os.path.join(full_dir, sub_dir2)

            for img_file in os.listdir(full_dir2):
                full_path_name = os.path.join(full_dir2, img_file)
                if os.path.isfile(full_path_name) and img_file.lower().endswith(tuple(ext_filter)):
                    ori_img = skio.imread(full_path_name)
                    gry_img = skcolor.rgb2gray(ori_img)
                    res_img = enhance_contrast(gry_img, disk(5))
                    # res_img = exps.adjust_gamma(gry_img)
                    # res2_img = exps.rescale_intensity(res_img)
                    # flt_img = np.array(res_img > flts.threshold_li(res_img))
                    res_imgs.append(res_img.reshape(-1))

    arr_imgs = np.asarray(res_imgs)
    img_cnt = arr_imgs.shape[0]
    n_col = math.ceil(math.sqrt(img_cnt))
    n_row = math.ceil(img_cnt / n_col)

    fig = plt.figure(figsize=(18, 12))
    for i in range(0, img_cnt):
        ax = fig.add_subplot(n_row, n_col, (i + 1))
        img = ax.imshow(arr_imgs[i].reshape(28, 96))

        img.axes.get_xaxis().set_visible(False)
        img.axes.get_yaxis().set_visible(False)

    plt.suptitle('All {} Samples'.format(img_cnt), size=12, x=0.518, y=0.935)
    plt.show()