import cv2
import json

from textdect.yolonet import YoloNet


def predict(config_file, weights_path_file, predict_file_list):
    with open(config_file) as config_buffer:
        config = json.loads(config_buffer.read())

    yolo_net = YoloNet(config)
    yolo_net.load_weights(weights_path_file)

    for img_file in predict_file_list:
        image = cv2.imread(img_file)
        boxes = yolo_net.predict(image)
        image = draw_boxes(image, boxes)

        head, tail = os.path.split(img_file)
        cv2.imwrite(os.path.join(head, 'predicted_{}'.format(tail)), image)


def draw_boxes(image, boxes):
    print("box len = {}".format(len(boxes)))
    for box in boxes:
        xmin, ymin, xmax, ymax = box.get_coordinates()
        print("{}, {}, {}, {}".format(xmin, ymin, xmax, ymax))
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)

    return image

if __name__ == '__main__':
    import os
    from settings import PROJECT_ROOT

    weights_file = os.path.join(PROJECT_ROOT, 'Data', 'Result', 'weights.h5')
    test_dir = os.path.join(PROJECT_ROOT, 'Data', 'Step1', 'Test')
    test_file_list = []
    for ii in range(1, 45):
        test_file = os.path.join(test_dir, 'sign{}.jpg'.format(ii))
        test_file_list.append(test_file)

    predict('config.json', weights_file, test_file_list)