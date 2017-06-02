import math
import matplotlib.pyplot as plt
import numpy as np
import os

from skimage import color as skcolor
from skimage import exposure as exps
from skimage import filters as flts
from skimage import transform as tsf
from skimage.morphology import disk
from skimage.filters.rank import enhance_contrast
from skimage import io as skio

from settings import PROJECT_ROOT

CLS_IMG_HEIGHT = 28
CLS_IMG_WIDTH = 96

EXT_FILTER = ['.jpg', '.png']


def get_immediate_subfolders(input_dir):
    return [folder_name for folder_name in os.listdir(input_dir)
            if os.path.isdir(os.path.join(input_dir, folder_name))]


def resize_to_desired(input_img):
    return tsf.resize(input_img, (CLS_IMG_HEIGHT, CLS_IMG_WIDTH), mode='constant')


# This takes the positive training/validation data from step 1 and generates the training/validation
# data for step 2.
def resize_images_from_step1():
    # Copy all images into this folder: $PROJECT_ROOT/Data/Step2/Temp/Images/InThisFolder/,
    # which will be placed along with the 'Resized' folder. Then run this script. You will
    # get all the resized images in the resized folder.
    in_dir = os.path.join(PROJECT_ROOT, 'Data', 'Step2', 'Temp')

    for sub_dir in get_immediate_subfolders(in_dir):
        full_dir = os.path.join(in_dir, sub_dir)
        for sub_dir2 in get_immediate_subfolders(full_dir):
            full_dir2 = os.path.join(full_dir, sub_dir2)
            resized_dir = os.path.join(full_dir2, 'Resized')

            for img_file in os.listdir(full_dir2):
                full_path_name = os.path.join(full_dir2, img_file)
                if os.path.isfile(full_path_name) and img_file.lower().endswith(tuple(EXT_FILTER)):
                    ori_img = skio.imread(full_path_name)
                    new_img = resize_to_desired(ori_img)
                    new_path_name = os.path.join(full_dir2, 'Resized', img_file)
                    skio.imsave(new_path_name, new_img)


# This takes the files stored in OTM and rename them into one single folder for future usage.
def rename_image_files_from_source(input_dir, output_dir):
    for sub_dir in get_immediate_subfolders(input_dir):
        full_dir = os.path.join(input_dir, sub_dir)
        for img_file in os.listdir(full_dir):
            full_path_name = os.path.join(full_dir, img_file)
            if os.path.isfile(full_path_name) and \
                    img_file.lower().endswith(tuple(EXT_FILTER)) and \
                            os.path.getsize(full_path_name) > 0:
                out_name = 's_' + sub_dir + '_' + img_file
                out_path_name = os.path.join(output_dir, out_name)
                os.rename(full_path_name, out_path_name)


# This is employed to compare the performance of difference filtering mechanisms offered in
# skimage package.
def filter_test():
    in_dir = os.path.join(PROJECT_ROOT, 'Data', 'Step2', 'Validation')

    res_imgs = []
    for sub_dir in get_immediate_subfolders(in_dir):
        full_dir = os.path.join(in_dir, sub_dir)
        for sub_dir2 in get_immediate_subfolders(full_dir):
            full_dir2 = os.path.join(full_dir, sub_dir2)

            for img_file in os.listdir(full_dir2):
                full_path_name = os.path.join(full_dir2, img_file)
                if os.path.isfile(full_path_name) and img_file.lower().endswith(tuple(EXT_FILTER)):
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

if __name__ == "__main__":
    action = 'Rename'

    if action == 'Resize':
        resize_images_from_step1()
    elif action == 'Rename':
        in_dir = os.path.join(PROJECT_ROOT, 'Temp', '05')
        out_dir = os.path.join(PROJECT_ROOT, 'Temp', 'New')

        rename_image_files_from_source(input_dir=in_dir, output_dir=out_dir)
    elif action == 'Filter':
        filter_test()
