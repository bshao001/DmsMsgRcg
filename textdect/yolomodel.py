import tensorflow as tf
from tensorflow import keras


class FullYolo(object):
    def __init__(self, img_height, img_width, grid_h, grid_w):
        # The function to implement the organization layer (thanks to github.com/allanzelener/YAD2K)
        def space_to_depth_x2(x):
            return tf.space_to_depth(x, block_size=2)

        input_image = keras.layers.Input(shape=(img_height, img_width, 3))

        # Layer 1
        x = keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same', name='conv_1', use_bias=False)(input_image)
        x = keras.layers.BatchNormalization(name='norm_1')(x)
        x = keras.layers.LeakyReLU(alpha=0.1)(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 2
        x = keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='conv_2', use_bias=False)(x)
        x = keras.layers.BatchNormalization(name='norm_2')(x)
        x = keras.layers.LeakyReLU(alpha=0.1)(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 3
        x = keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='conv_3', use_bias=False)(x)
        x = keras.layers.BatchNormalization(name='norm_3')(x)
        x = keras.layers.LeakyReLU(alpha=0.1)(x)

        # Layer 4
        x = keras.layers.Conv2D(64, (1, 1), strides=(1, 1), padding='same', name='conv_4', use_bias=False)(x)
        x = keras.layers.BatchNormalization(name='norm_4')(x)
        x = keras.layers.LeakyReLU(alpha=0.1)(x)

        # Layer 5
        x = keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='conv_5', use_bias=False)(x)
        x = keras.layers.BatchNormalization(name='norm_5')(x)
        x = keras.layers.LeakyReLU(alpha=0.1)(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 6
        x = keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv_6', use_bias=False)(x)
        x = keras.layers.BatchNormalization(name='norm_6')(x)
        x = keras.layers.LeakyReLU(alpha=0.1)(x)

        # Layer 7
        x = keras.layers.Conv2D(128, (1, 1), strides=(1, 1), padding='same', name='conv_7', use_bias=False)(x)
        x = keras.layers.BatchNormalization(name='norm_7')(x)
        x = keras.layers.LeakyReLU(alpha=0.1)(x)

        # Layer 8
        x = keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv_8', use_bias=False)(x)
        x = keras.layers.BatchNormalization(name='norm_8')(x)
        x = keras.layers.LeakyReLU(alpha=0.1)(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 9
        x = keras.layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_9', use_bias=False)(x)
        x = keras.layers.BatchNormalization(name='norm_9')(x)
        x = keras.layers.LeakyReLU(alpha=0.1)(x)

        # Layer 10
        x = keras.layers.Conv2D(256, (1, 1), strides=(1, 1), padding='same', name='conv_10', use_bias=False)(x)
        x = keras.layers.BatchNormalization(name='norm_10')(x)
        x = keras.layers.LeakyReLU(alpha=0.1)(x)

        # Layer 11
        x = keras.layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_11', use_bias=False)(x)
        x = keras.layers.BatchNormalization(name='norm_11')(x)
        x = keras.layers.LeakyReLU(alpha=0.1)(x)

        # Layer 12
        x = keras.layers.Conv2D(256, (1, 1), strides=(1, 1), padding='same', name='conv_12', use_bias=False)(x)
        x = keras.layers.BatchNormalization(name='norm_12')(x)
        x = keras.layers.LeakyReLU(alpha=0.1)(x)

        # Layer 13
        x = keras.layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_13', use_bias=False)(x)
        x = keras.layers.BatchNormalization(name='norm_13')(x)
        x = keras.layers.LeakyReLU(alpha=0.1)(x)

        skip_connection = x

        x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 14
        x = keras.layers.Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_14', use_bias=False)(x)
        x = keras.layers.BatchNormalization(name='norm_14')(x)
        x = keras.layers.LeakyReLU(alpha=0.1)(x)

        # Layer 15
        x = keras.layers.Conv2D(512, (1, 1), strides=(1, 1), padding='same', name='conv_15', use_bias=False)(x)
        x = keras.layers.BatchNormalization(name='norm_15')(x)
        x = keras.layers.LeakyReLU(alpha=0.1)(x)

        # Layer 16
        x = keras.layers.Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_16', use_bias=False)(x)
        x = keras.layers.BatchNormalization(name='norm_16')(x)
        x = keras.layers.LeakyReLU(alpha=0.1)(x)

        # Layer 17
        x = keras.layers.Conv2D(512, (1, 1), strides=(1, 1), padding='same', name='conv_17', use_bias=False)(x)
        x = keras.layers.BatchNormalization(name='norm_17')(x)
        x = keras.layers.LeakyReLU(alpha=0.1)(x)

        # Layer 18
        x = keras.layers.Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_18', use_bias=False)(x)
        x = keras.layers.BatchNormalization(name='norm_18')(x)
        x = keras.layers.LeakyReLU(alpha=0.1)(x)

        # Layer 19
        x = keras.layers.Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_19', use_bias=False)(x)
        x = keras.layers.BatchNormalization(name='norm_19')(x)
        x = keras.layers.LeakyReLU(alpha=0.1)(x)

        # Layer 20
        x = keras.layers.Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_20', use_bias=False)(x)
        x = keras.layers.BatchNormalization(name='norm_20')(x)
        x = keras.layers.LeakyReLU(alpha=0.1)(x)

        # Layer 21
        skip_connection = keras.layers.Conv2D(64, (1, 1), strides=(1, 1), padding='same', name='conv_21',
                                              use_bias=False)(skip_connection)
        skip_connection = keras.layers.BatchNormalization(name='norm_21')(skip_connection)
        skip_connection = keras.layers.LeakyReLU(alpha=0.1)(skip_connection)
        skip_connection = keras.layers.Lambda(space_to_depth_x2)(skip_connection)

        x = keras.layers.concatenate([skip_connection, x])

        # Layer 23
        x = keras.layers.Conv2D((1 + 4), (1, 1), strides=(1, 1), padding='same', name='conv_23')(x)
        output = keras.layers.Reshape((grid_h, grid_w, (1 + 4)))(x)

        self.model = keras.models.Model(input_image, output)


class TinyYolo(object):
    def __init__(self, img_height, img_width, grid_h, grid_w):
        input_image = keras.layers.Input(shape=(img_height, img_width, 3))

        # Layer 1
        x = keras.layers.Conv2D(16, (3,3), strides=(1,1), padding='same', name='conv_1',
                                use_bias=False)(input_image)
        x = keras.layers.BatchNormalization(name='norm_1')(x)
        x = keras.layers.LeakyReLU(alpha=0.1)(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 2 - 5
        for i in range(0,4):
            x = keras.layers.Conv2D(32*(2**i), (3,3), strides=(1,1), padding='same',
                                    name='conv_' + str(i+2), use_bias=False)(x)
            x = keras.layers.BatchNormalization(name='norm_' + str(i+2))(x)
            x = keras.layers.LeakyReLU(alpha=0.1)(x)
            x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 6
        x = keras.layers.Conv2D(512, (3,3), strides=(1,1), padding='same',
                                name='conv_6', use_bias=False)(x)
        x = keras.layers.BatchNormalization(name='norm_6')(x)
        x = keras.layers.LeakyReLU(alpha=0.1)(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(x)

        # Layer 7 - 8
        for i in range(0,2):
            x = keras.layers.Conv2D(1024, (3,3), strides=(1,1), padding='same',
                                    name='conv_' + str(i+7), use_bias=False)(x)
            x = keras.layers.BatchNormalization(name='norm_' + str(i+7))(x)
            x = keras.layers.LeakyReLU(alpha=0.1)(x)

        # Layer 9
        x = keras.layers.Conv2D((4 + 1), (1, 1), strides=(1, 1), padding='same', name='conv_9')(x)
        output = keras.layers.Reshape((grid_h, grid_w, (4 + 1)))(x)

        self.model = keras.models.Model(input_image, output)
