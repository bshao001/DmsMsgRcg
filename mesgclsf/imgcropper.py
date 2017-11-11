import tensorflow as tf
import os
from skimage import color as skcolor
from skimage import io as skio

from settings import PROJECT_ROOT
from misc.cnnpredictor import CnnPredictor
from textdect.textdetector import detect


def crop_detected_areas(detector, input_dir, input_file, output_dir):
    input_full_name = os.path.join(input_dir, input_file)
    img_arr = skio.imread(input_full_name)
    gry_arr = skcolor.rgb2gray(img_arr)
    areas, _ = detect(detector, gry_arr)

    i = 1
    for area in areas:
        y, x, h, w = area[0], area[1], area[2], area[3]
        x2, y2 = x + w, y + h

        area_img = img_arr[y:y2, x:x2]
        out_file = input_file[:-4] + '_' + str(i) + input_file[-4:]
        out_full_name = os.path.join(output_dir, out_file)
        skio.imsave(out_full_name, area_img)
        i += 1

if __name__ == "__main__":
    ext_filter = ['.jpg', '.png']
    in_dir = os.path.join(PROJECT_ROOT, 'Data', 'Temp', 'OtmImages')
    out_dir = os.path.join(PROJECT_ROOT, 'Data', 'Temp', 'CroppedImages')

    model_dir = os.path.join(PROJECT_ROOT, 'Data', 'Result')

    with tf.Session() as sess:
        det = CnnPredictor(sess, 's1', model_dir, 'step1_basic_s1')

        cnt = 0
        for img_file in os.listdir(in_dir):
            full_path_name = os.path.join(in_dir, img_file)
            if os.path.isfile(full_path_name) and img_file.lower().endswith(tuple(ext_filter)):
                if cnt > 0 and cnt % 100 == 0:
                    print("{} images cropped".format(cnt))
                crop_detected_areas(det, in_dir, img_file, out_dir)
                cnt += 1

        print("Altogether, {} images cropped".format(cnt))