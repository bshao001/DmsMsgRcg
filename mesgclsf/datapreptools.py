import cv2
import math
import os

CLS_IMG_HEIGHT = 28
CLS_IMG_WIDTH = 96


def resize_to_desired(input_img):
    h = input_img.shape[0]
    if h < 20:  # pad the image to 20 or 21 pixels height if it is too short
        border = math.ceil((20 - h) / 2)
        new_img = cv2.copyMakeBorder(input_img, top=border, bottom=border, left=0, right=0,
                                     borderType=cv2.BORDER_DEFAULT)
    else:
        new_img = input_img

    return cv2.resize(new_img, (CLS_IMG_WIDTH, CLS_IMG_HEIGHT))


if __name__ == "__main__":
    # This script prepares training samples for step 2 based on the manual labels in step 1.
    from settings import PROJECT_ROOT

    input_dir = os.path.join(PROJECT_ROOT, 'Data', 'Step1', 'Training', 'AntImages')
    output_dir = os.path.join(PROJECT_ROOT, 'Data', 'Temp', 'ResizedImages')
    label_file = os.path.join(PROJECT_ROOT, 'Data', 'Step1', 'Training', 'labels.txt')

    with open(label_file, 'r') as label_f:
        for line in label_f:
            ln = line.strip()
            if not ln:
                continue
            s = ln.split(';')
            if len(s) == 1:
                continue
            img_file = s[0].strip()
            full_img = cv2.imread(os.path.join(input_dir, img_file))

            for i in range(1, len(s)):
                xmin, ymin, xmax, ymax = s[i].strip()[1:-1].split(',')
                area_img = full_img[int(ymin):int(ymax), int(xmin):int(xmax), :]
                resized_img = resize_to_desired(area_img)
                out_file = img_file[:-4] + '_' + str(i) + '.png'  # Always save as png files
                out_full_name = os.path.join(output_dir, out_file)
                cv2.imwrite(out_full_name, resized_img)
