import math
import numpy as np
import os
from skimage import color as skcolor
from skimage import io as skio
from skimage import transform as tsf


class ImgReader(object):
    """
    This reader is to read input image(s) and convert that into an array of image features.
    """
    def __init__(self, feature_height, feature_width):
        self.feature_height = feature_height
        self.feature_width = feature_width
        self.feature_count = feature_height * feature_width

    def get_features_all_images(self, img_dir, ext_filter=['.jpg', '.png'], skip=[0, 0, 0, 0],
                                stride=5, padding=True, data_augm=False):
        """
        Output the features extracted from all images in one folder. This method is designed only
            for the trainers.
        Args:
            img_dir: The full path to the images to be feature-extracted.
            ext_filter: Optional. File name filter.
            skip: Optional. The pixels to be ignored in the border areas in each side, assuming no
                useful information is contained in these areas. Fixed to be 4 numbers, in the order
                of top, right, bottom, and left.
            stride: Optional. The stride of the sliding.
            padding: Optional. Whether to pad the image to fit the feature space size or to
                discard the extra pixels if padding is False.
            data_augm: Optional. Whether to perform data augmentation for the given image. The
                only data augmentation approach applied here is to rotate 20 degree clockwise
                and rotate 20 degree anti-clockwise so that 1 image becomes 3 images.
        Returns:
            A matrix (python list), in which each row contains the features of the sampling sliding
            window, while the number of rows depends on the number of the images in the given folder
            and the image size of the input, and other parameters.
        """
        all_features = []
        for img_file in os.listdir(img_dir):
            full_path_name = os.path.join(img_dir, img_file)
            if os.path.isfile(full_path_name) and img_file.lower().endswith(tuple(ext_filter)):
                img = skio.imread(full_path_name)
                img_arr = skcolor.rgb2gray(img)
                _, features = self.get_image_array_features(img_arr, skip, stride, padding)
                if len(features) > 0:
                    all_features.extend(features)

                if data_augm:
                    left_arr = tsf.rotate(img_arr, -25)
                    _, left_feats = self.get_image_array_features(left_arr, skip, stride, padding)
                    if len(left_feats) > 0:
                        all_features.extend(left_feats)

                    right_arr = tsf.rotate(img_arr, 25)
                    _, right_feats = self.get_image_array_features(right_arr, skip, stride, padding)
                    if len(right_feats) > 0:
                        all_features.extend(right_feats)

        return all_features

    def get_image_features(self, img_file, skip=[0, 0, 0, 0], stride=5, padding=True):
        """
        Take an image file as input, and output an array of image features whose matrix size is
        based on the image size. When no padding, and the image size is smaller than the required
        feature space size (in x or y direction), the image is not checked, and this method will
        return a tuple of two empty lists; When padding is True, and the image size is more than
        4 pixels smaller than the require feature space size (in x or y direction), the image is
        not checked either. This method can be used by both the trainer and predictor.
        Args:
            img_file: The file name of the image.
            skip: Optional. The pixels to be ignored in the border areas in each side, assuming
                no useful information is contained in these areas. Fixed to be 4 numbers, in the
                order of top, right, bottom, and left.
            stride: Optional. The stride of the sliding.
            padding: Optional. Whether to pad the image to fit the feature space size or to
                discard the extra pixels if padding is False.
        Returns:
            coordinates: A list of coordinates, each of which contains y and x that are the top
                left corner offsets of the sliding window.
            features: A matrix (python list), in which each row contains the features of the
                sampling sliding window, while the number of rows depends on the image size of
                the input.
        """
        img = skio.imread(img_file)
        img_arr = skcolor.rgb2gray(img)

        return self.get_image_array_features(img_arr, skip, stride, padding)

    def get_image_array_features(self, img_arr, skip=[0, 0, 0, 0], stride=5, padding=True):
        """
        Take an image file as input, and output an array of image features whose matrix size is
        based on the image size. When no padding, and the image size is smaller than the required
        feature space size (in x or y direction), the image is not checked, and this method will
        return a tuple of two empty lists; When padding is True, and the image size is more than
        4 pixels smaller than the require feature space size (in x or y direction), the image is
        not checked either. This method can be used by both the trainer and predictor.
        Note that when stride is greater than 5, padding is not supported, and it will be reset
        to False regardless of the input.
        Args:
            img_arr: The image array (a numpy ndarray) read from the image file. It has already
                been changed to gray scale.
            skip: Optional. The pixels to be ignored in the border areas in each side, assuming
                no useful information is contained in these areas. Fixed to be 4 numbers, in the
                order of top, right, bottom, and left.
            stride: Optional. The stride of the sliding.
            padding: Optional. Whether to pad the image to fit the feature space size or to
                discard the extra pixels if padding is False.
        Returns:
            coordinates: A list of coordinates, each of which contains y and x that are the top
                left corner offsets of the sliding window.
            features: A matrix (python list), in which each row contains the features of the
                sampling sliding window, while the number of rows depends on the image size of
                the input.
        """
        assert stride >= 1
        if stride > 5:
            padding = False

        coordinates, features = [], []  # two lists to be returned

        img_height, img_width = img_arr.shape

        if skip[0] > 0 or skip[1] > 0 or skip[2] > 0 or skip[3] > 0:
            img_arr = img_arr[skip[0]:img_height-skip[2], skip[3]:img_width-skip[1]]
            img_height, img_width = img_arr.shape

        padding_top, padding_left = 0, 0

        if not padding:
            if img_height < self.feature_height or img_width < self.feature_width:
                print("Image with size: {}x{} is too small. Ignored in when no padding."
                      .format(img_width, img_height))
                return coordinates, features
        else:
            if img_height+4 < self.feature_height or img_width+4 < self.feature_width:
                print("Image with size: {}x{} is too small. Ignored in padding mode."
                      .format(img_width, img_height))
                return coordinates, features

            if img_height > self.feature_height:
                extra_y = (img_height - self.feature_height) % stride
                if extra_y > 0:
                    padding_y = stride - extra_y
                else:
                    padding_y = 0
            elif img_height < self.feature_height:
                padding_y = self.feature_height - img_height
            else:
                padding_y = 0

            if img_width > self.feature_width:
                extra_x = (img_width - self.feature_width) % stride
                if extra_x > 0:
                    padding_x = stride - extra_x
                else:
                    padding_x = 0
            elif img_width < self.feature_width:
                padding_x = self.feature_width - img_width
            else:
                padding_x = 0

            if padding_y > 0 or padding_x > 0:
                padding_top = math.floor(padding_y / 2)
                padding_left = math.floor(padding_x / 2)

                new_y, new_x = img_height + padding_y, img_width + padding_x
                new_img = np.zeros((new_y, new_x))
                new_img[padding_top:padding_top+img_height,
                        padding_left:padding_left+img_width]=img_arr
                img_arr = new_img
                img_height, img_width = img_arr.shape

        for y in range(0, img_height-self.feature_height+1, stride):
            for x in range(0, img_width-self.feature_width+1, stride):
                orig_x = x + skip[3] - padding_left
                orig_y = y + skip[0] - padding_top
                coordinates.append((orig_y, orig_x))
                this_win = img_arr[y:y+self.feature_height, x:x+self.feature_width]
                features.append(this_win.reshape(-1))

        return coordinates, features

if __name__ == "__main__":
    import os
    from misc.imgtools import plot_samples
    from settings import PROJECT_ROOT

    img_reader = ImgReader(28, 28)

    img_dr = os.path.join(PROJECT_ROOT, 'Data', 'Training', 'tasMsg', 'positive')
    all_feats = img_reader.get_features_all_images(img_dr, stride=3)

    X_features = np.asarray(all_feats)
    print("X_features size: {}".format(X_features.shape))

    plot_samples(X_features, 28, 28, transpose=False, shuffle=True)