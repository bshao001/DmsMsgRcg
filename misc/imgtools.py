import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from skimage import transform as tsf
from skimage import filters as flt


def plot_samples(X_images, img_height, img_width, figsize=(5, 5), transpose=True,
                 shuffle=True):
    """
    Args:
        X_images: A 2-D ndarray (matrix) each row of which holds the pixels as features
            of one image. The row number will be the number of all input images.
        img_height: The pixel numbers of the input image in height.
        img_width: The pixel numbers of the input image in width.
        figsize: Optional. The size of each small figure.
        transpose: Optional. Whether to transpose the image array. When the image attributes
            come from matlab, it needs to be transposed by default.
        shuffle: Optional. Whether to shuffle the input array.
    """
    img_cnt, feature_cnt = X_images.shape
    assert feature_cnt == img_height * img_width

    if (shuffle):
        images = np.random.permutation(X_images)
    else:
        images = X_images

    if img_cnt >= 100:
        n_row, n_col, samp_cnt = 10, 10, 100
    elif img_cnt >= 64:
        n_row, n_col, samp_cnt = 8, 8, 64
    else:
        n_row, n_col, samp_cnt = 0, 0, 0

    if img_cnt >= samp_cnt > 0:
        samps = images[0: samp_cnt]

        plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(n_row, n_col, wspace=0.0, hspace=0.0)

        for i in range(0, n_row):
            for j in range(0, n_col):
                ax = plt.subplot(gs[i, j])
                idx = i * n_col + j;
                img = samps[idx].reshape(img_height, img_width)
                if transpose:
                    img = img.T
                fig = ax.imshow(img, interpolation='nearest')
                fig.axes.get_xaxis().set_visible(False)
                fig.axes.get_yaxis().set_visible(False)

        plt.suptitle('{} out of {} Samples'.format(samp_cnt, img_cnt), size=12, x=0.515, y=0.935)
        plt.show()
    else:
        samps = images

        n_col = math.ceil(math.sqrt(img_cnt))
        n_row = math.ceil(img_cnt / n_col)

        fig = plt.figure(figsize=figsize)
        for i in range(0, img_cnt):
            ax = fig.add_subplot(n_row, n_col, (i + 1))
            if transpose:
                img = ax.imshow(samps[i].reshape(img_height, img_width).T)
            else:
                img = ax.imshow(samps[i].reshape(img_height, img_width))

            img.axes.get_xaxis().set_visible(False)
            img.axes.get_yaxis().set_visible(False)

        plt.suptitle('All {} Samples'.format(img_cnt), size=12, x=0.518, y=0.935)
        plt.show()


def rotated_images(X_images, img_height, img_width, angle):
    """
    Args:
        X_images: A 2-D ndarray (matrix) each row of which holds the pixels as features
            of one image. The row number will be the number of all input images.
        img_height: The pixel numbers of the input image in height.
        img_width: The pixel numbers of the input image in width.
        angle: Rotation angle in degrees in counter-clockwise direction. Refer to
            skimage.transform.rotate for details.
    Returns: The same 2-D ndarray (matrix) each row of which holds the pixels as features
            of one image, except that, each image has been rotated in angle degree.
    """
    img_cnt = X_images.shape[0]
    reshaped_images = np.reshape(X_images.T, (img_height, img_width, img_cnt))
    tmp = tsf.rotate(reshaped_images, angle)
    return np.reshape(tmp, (img_height*img_width, img_cnt)).T


def gaussian_filtered_images(X_images, img_height, img_width, sigma):
    """
    Args:
        X_images: A 2-D ndarray (matrix) each row of which holds the pixels as features
            of one image. The row number will be the number of all input images.
        img_height: The pixel numbers of the input image in height.
        img_width: The pixel numbers of the input image in width.
        sigma: A scalar or sequence of scalars representing standard deviation for Gaussian
            kernel. Refer to skimage.filters.gaussian for details.
    Returns: The same 2-D ndarray (matrix) each row of which holds the pixels as features
            of one image, except that, each image has been filtered.
    """
    img_cnt = X_images.shape[0]
    reshaped_images = np.reshape(X_images.T, (img_height, img_width, img_cnt))
    filtered_images = flt.gaussian(reshaped_images, sigma)
    return np.reshape(filtered_images, (img_height*img_width, img_cnt)).T


def sobel_filtered_images(X_images, img_height, img_width):
    """
    Args:
        X_images: A 2-D ndarray (matrix) each row of which holds the pixels as features
            of one image. The row number will be the number of all input images.
        img_height: The pixel numbers of the input image in height.
        img_width: The pixel numbers of the input image in width.
    Returns: The same 2-D ndarray (matrix) each row of which holds the pixels as features
            of one image, except that, each image has been filtered.
    """
    img_cnt, _ = X_images.shape
    out_images = []
    for i in range(img_cnt):
        reshaped_image = np.reshape(X_images[i], (img_height, img_width))
        filtered_image = flt.sobel(reshaped_image)
        out_images.append(filtered_image.reshape(-1))

    return np.asarray(out_images)


def threshold_filtered_images(X_images, img_height, img_width):
    """
    Args:
        X_images: A 2-D ndarray (matrix) each row of which holds the pixels as features
            of one image. The row number will be the number of all input images.
        img_height: The pixel numbers of the input image in height.
        img_width: The pixel numbers of the input image in width.
    Returns: The same 2-D ndarray (matrix) each row of which holds the pixels as features
            of one image, except that, each image has been filtered.
    """
    img_cnt, _ = X_images.shape
    out_images = []
    for i in range(img_cnt):
        reshaped_image = np.reshape(X_images[i], (img_height, img_width))
        filtered_image = np.array(reshaped_image > flt.threshold_li(reshaped_image))
        out_images.append(filtered_image.reshape(-1))

    return np.asarray(out_images)

if __name__ == "__main__":
    import os
    from scipy import io as spio
    from settings import PROJECT_ROOT

    mat_data_file = os.path.join(PROJECT_ROOT, 'Data', 'Step4', 'Training', 'ex4data1.mat')
    data_dict = spio.loadmat(mat_data_file)
    X = data_dict['X']
    flt_X = threshold_filtered_images(X, 20, 20)

    plot_samples(flt_X, 20, 20, shuffle=True)