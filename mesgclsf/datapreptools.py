import os

from skimage import transform as tsf
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
