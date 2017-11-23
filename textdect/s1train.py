import json
import os

from textdect.yolonet import YoloNet


def train(config_file, train_image_dir, train_label_file,  weights_path_file, train_log_dir):
    with open(config_file) as config_buffer:
        config = json.loads(config_buffer.read())

    file_list = []
    for img_file in os.listdir(train_image_dir):
        full_path_name = os.path.join(img_dir, img_file)
        if os.path.isfile(full_path_name) and img_file.lower().endswith(tuple(['.jpg', '.png'])):
            file_list.append(img_file)

    train_data = read_image_data(file_list, train_label_file)
    print("# Train data size: {}".format(len(train_data)))

    yolo_net = YoloNet(config)
    yolo_net.train(train_image_dir, train_data,  weights_path_file, train_log_dir)


def read_image_data(image_file_list, label_path_file):
    """
    A sample line in the label file:
        aa.jpg; [100, 120, 200, 156]; [100, 200, 208, 248]
    """
    image_data = []

    with open(label_path_file, 'r') as label_f:
        for line in label_f:
            ln = line.strip()
            if not ln:
                continue
            img_item = {}
            s = ln.split(';')
            if len(s) == 1 or s[0].strip() not in image_file_list:
                continue
            img_item['filename'] = s[0].strip()

            tmp = []
            for i in range(1, len(s)):
                xmin, ymin, xmax, ymax = s[i].strip()[1:-1].split(',')
                tmp.append((xmin, ymin, xmax, ymax))

            img_item['labels'] = tmp

            image_data.append(img_item)

    return image_data


if __name__ == '__main__':
    from settings import PROJECT_ROOT

    img_dir = os.path.join(PROJECT_ROOT, 'Data', 'AntImages')

    label_file = os.path.join(PROJECT_ROOT, 'Data', 'label.txt')
    log_dir = os.path.join(PROJECT_ROOT, 'Data', 'Result', 'Logs')
    weights_file = os.path.join(PROJECT_ROOT, 'Data', 'Result', 'weights.{epoch:02d}.h5')

    train('config.json', img_dir, label_file,  weights_file, log_dir)
