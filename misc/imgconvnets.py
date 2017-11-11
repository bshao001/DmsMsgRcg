import math
import numpy as np
import os
import tensorflow as tf

from misc.spatial_transformer import transformer


class ImgConvNets(object):
    """
    This ConvNets is designed to have fixed layers, with a few model options selectable,
    for image recognition applications. All options require that image height and width
    are multiplications of 4.
    """
    def __init__(self, model, model_scope, img_height, img_width, class_count, keep_prob=0.5,
                 learning_rate=1e-4, lr_adaptive=True, batch_size=32, max_steps=20000):
        """
        Args:
            model: Specify which model to use.
            model_scope: The variable_scope used to separate this meta graph from other
                meta graph when multiple models are restored into the same graph.
            img_height: The pixel numbers of the input image in height.
            img_width: The pixel numbers of the input image in width.
            class_count: Number of the output classes.
            keep_prob: optional. The probability a neuron's output is kept during dropout.
            learning_rate: optional. The learning rate for the optimization.
            lr_adaptive: optional. Whether to adjust the learning rate based on the training
                accuracy. If True, the given learning_rate will be ignored.
            batch_size: optional. The number of samples to be used in one step of the
                optimization process.
            max_steps: optional. The max number of iterative steps in the training process.
        """
        assert model == 'BASIC' or model == 'DCNN' or model == 'STCNN'

        self.model = model
        self.model_scope = model_scope
        self.img_height = img_height
        self.img_width = img_width
        self.class_count = class_count
        self.keep_prob = keep_prob
        self.learning_rate = learning_rate
        self.lr_adaptive = lr_adaptive
        self.batch_size = batch_size
        self.max_steps = max_steps

    def train(self, img_features, true_labels, train_dir, result_file):
        """
        Note that img_height * img_width must match the column size of the img_features. The
        training result is saved in files including logits named logits; placeholders named:
        images, labels, and keep_prob; and other parameters: train_op, loss, accuracy, which
        can be later retrieved the collection for further training or prediction.
        Args:
            img_features: A 2-D ndarray (matrix) each row of which holds the pixels as
                features of one image. The row number will be the training sample size.
            true_labels: The true labels of the training samples.
            train_dir: The full path to the folder in which the result_file locates.
            result_file: The file name to save the train result.
        """
        rows, cols = img_features.shape
        if cols != self.img_height * self.img_width:
            raise ValueError("Image feature dimension does not match the given "
                             "image size parameters")

        train_set = np.random.permutation(np.append(img_features, true_labels, axis=1))

        def_graph = tf.Graph()
        with def_graph.as_default():
            with tf.variable_scope(self.model_scope):
                images_placeholder = tf.placeholder(tf.float32, name='images_placeholder')
                labels_placeholder = tf.placeholder(tf.int32)

                keep_prob_placeholder = tf.placeholder(tf.float32, name='keep_prob_placeholder')
                learning_rate_placeholder = tf.placeholder(tf.float32, shape=[])

                # Build a Graph that computes forward prop from the inference model.
                if self.model == 'STCNN':
                    logits = self._build_inference_graph_stcnn(images_placeholder, keep_prob_placeholder)
                elif self.model == 'DCNN':
                    logits = self._build_inference_graph_dcnn(images_placeholder, keep_prob_placeholder)
                else:
                    logits = self._build_inference_graph_basic(images_placeholder, keep_prob_placeholder)

            # Save the variables within the model_scope with given model_scope as prefix.
            tf.add_to_collection(self.model_scope+"images", images_placeholder)
            tf.add_to_collection(self.model_scope+"labels", labels_placeholder)

            tf.add_to_collection(self.model_scope+"keep_prob", keep_prob_placeholder)
            tf.add_to_collection(self.model_scope+"learning_rate", learning_rate_placeholder)
            tf.add_to_collection(self.model_scope+"logits", logits)

            # Add to the Graph the Ops that calculate and apply gradients.
            train_op, loss, accuracy = \
                self._build_training_graph(logits, labels_placeholder, learning_rate_placeholder)

            tf.add_to_collection(self.model_scope+"train_op", train_op)
            tf.add_to_collection(self.model_scope+"loss", loss)
            tf.add_to_collection(self.model_scope+"accuracy", accuracy)

            # Create a saver for writing training results.
            saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                     scope=self.model_scope), max_to_keep=10)

        with tf.Session(graph=def_graph) as sess:
            sess.run(tf.global_variables_initializer())

            # Start the training loop.
            loss_list = []
            accu_list = []
            last_accu = 0.0

            save_file = os.path.join(train_dir, result_file)

            disp_step = self._get_epoch_step_count(train_set.shape[0])
            for step in range(self.max_steps):
                # Read a batch of images and labels
                batch_data = self._get_next_batch(train_set, step*self.batch_size)
                images_feed, labels_feed = \
                    batch_data[:, :cols], batch_data[:, cols:].reshape(-1)

                lr_feed = self._get_learning_rate(last_accu)
                # Run one step of the model.  The return values are the activations
                # from the `train_op` (which is discarded) and the `loss` Op.
                _, loss_val, accu_val = sess.run([train_op, loss, accuracy],
                                                 feed_dict={images_placeholder: images_feed,
                                                            labels_placeholder: labels_feed,
                                                            learning_rate_placeholder: lr_feed,
                                                            keep_prob_placeholder: self.keep_prob})

                # Check to make sure the loss is decreasing
                loss_list.append(loss_val)
                accu_list.append(accu_val)
                if (step % disp_step == 0) or (step == self.max_steps-1):
                    mean_accu = sum(accu_list)*100/len(accu_list)
                    if mean_accu >= 99.68 and mean_accu > last_accu:
                        saver.save(sess, save_file, global_step=step)
                    elif step == self.max_steps - 1:
                        saver.save(sess, save_file)

                    print("Step {:6d}: learning_rate used = {:.6f}, average loss = {:7.4f}, "
                          "and training accuracy min = {:6.2f}%, mean = {:6.2f}%, "
                          "max = {:6.2f}%".format(step, lr_feed,
                                                  sum(loss_list)/len(loss_list),
                                                  min(accu_list)*100, mean_accu,
                                                  max(accu_list)*100))
                    if mean_accu >= 99.99: break

                    loss_list = []
                    accu_list = []
                    last_accu = mean_accu


    def retrain(self, img_features, true_labels, train_dir, last_file, new_file):
        """
        Note that img_height * img_width must match the column size of the img_features. The
        training result is saved in files including logits named logits, and placeholders named:
        images, labels, and keep_prob, which can be later retrieved the collection for further
        training or prediction.
        Args:
            img_features: A 2-D ndarray (matrix) each row of which holds the pixels as
                features of one image. The row number will be the training sample size.
            true_labels: The true labels of the training samples.
            train_dir: The full path to the folder in which the last_file and new_file locate.
            last_file: Name of the file that saved the training result of the previous training.
            new_file: The file name to save the train result.
        """
        rows, cols = img_features.shape
        if cols != self.img_height * self.img_width:
            raise ValueError("Image feature dimension does not match the given "
                             "image size parameters")

        train_set = np.random.permutation(np.append(img_features, true_labels, axis=1))

        with tf.Session(graph=tf.Graph()) as sess:
            rest_s = tf.train.import_meta_graph(os.path.join(train_dir, last_file + ".meta"))
            rest_s.restore(sess, os.path.join(train_dir, last_file))

            # Retrieve the Ops we 'remembered'.
            images_placeholder = tf.get_collection(self.model_scope+"images")[0]
            labels_placeholder = tf.get_collection(self.model_scope+"labels")[0]
            keep_prob_placeholder = tf.get_collection(self.model_scope+"keep_prob")[0]
            learning_rate_placeholder = tf.get_collection(self.model_scope+"learning_rate")[0]

            train_op = tf.get_collection(self.model_scope+"train_op")[0]
            loss = tf.get_collection(self.model_scope+"loss")[0]
            accuracy = tf.get_collection(self.model_scope+"accuracy")[0]

            # Prepare the variables for result output
            loss_list = []
            accu_list = []
            last_accu = 0.0

            saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                     scope=self.model_scope))
            save_file = os.path.join(train_dir, new_file)

            # Start the training loop.
            disp_step = self._get_epoch_step_count(train_set.shape[0])
            for step in range(self.max_steps):
                # Read a batch of images and labels
                batch_data = self._get_next_batch(train_set, step * self.batch_size)
                images_feed, labels_feed = \
                    batch_data[:, :cols], batch_data[:, cols:].reshape(-1)

                lr_feed = self._get_learning_rate(last_accu, retrain=True)
                # Run one step of the model.  The return values are the activations from
                # the train_op (which is discarded), the loss Op, and the accuracy
                _, loss_val, accu_val = sess.run([train_op, loss, accuracy],
                                                 feed_dict={images_placeholder: images_feed,
                                                            labels_placeholder: labels_feed,
                                                            learning_rate_placeholder: lr_feed,
                                                            keep_prob_placeholder: self.keep_prob})

                # Check to make sure the loss is decreasing
                loss_list.append(loss_val)
                accu_list.append(accu_val)
                if (step > 0 and step % disp_step == 0) or (step == self.max_steps - 1):
                    mean_accu = sum(accu_list) * 100 / len(accu_list)
                    if mean_accu >= 99.68 and mean_accu > last_accu:
                        saver.save(sess, save_file, global_step=step, write_meta_graph=False)
                    elif step == self.max_steps - 1:
                        saver.save(sess, save_file, write_meta_graph=False)

                    print("Step {:6d}: learning_rate used = {:.6f}, average loss = {:9.6f}, "
                          "and training accuracy min = {:6.2f}%, mean = {:6.2f}%, "
                          "max = {:6.2f}%".format(step, lr_feed,
                                                  sum(loss_list)/len(loss_list),
                                                  min(accu_list)*100, mean_accu,
                                                  max(accu_list)*100))
                    if step > 0 and mean_accu >= 99.99: break

                    loss_list = []
                    accu_list = []
                    last_accu = mean_accu

    def _build_inference_graph_stcnn(self, images, keep_prob):
        """
        Build initial inference graph.
        Args:
            images: Images placeholder.
            keep_prob: A placeholder for the probability that a neuron's output is kept
                    during dropout.
        Returns:
            logits: Output tensor with the computed logits.
        """
        # Transformer Layer
        with tf.name_scope('transformer'):
            shaped_images = tf.reshape(images, [-1, self.img_height, self.img_width, 1])

            # Define the two-layer localisation network, with a dropout layer
            # after the first layer.
            num_batch = 64

            W_fc_loc1 = tf.Variable(tf.zeros([self.img_height*self.img_width, num_batch]))
            b_fc_loc1 = tf.Variable(
                 tf.random_normal([num_batch], mean=0.0, stddev=0.01))
            h_fc_loc1 = tf.nn.tanh(tf.matmul(images, W_fc_loc1) + b_fc_loc1)

            h_fc_loc1_drop = tf.nn.dropout(h_fc_loc1, keep_prob)

            # Initialize the transformation to use identity matrix
            initial = np.array([[1., 0, 0], [0, 1., 0]], dtype='float32').flatten()

            W_fc_loc2 = tf.Variable(tf.zeros([num_batch, 6]))
            b_fc_loc2 = tf.Variable(initial_value=initial, name='b_fc_loc2')
            h_fc_loc2 = tf.nn.tanh(tf.matmul(h_fc_loc1_drop, W_fc_loc2) + b_fc_loc2)

            # Create a spatial transformer module to identify discriminative patches
            out_size = (self.img_height, self.img_width)
            h_trans = transformer(shaped_images, h_fc_loc2, out_size)
        # First Set of Convolutional Layers
        with tf.name_scope('conv1'):
            W_conv11 = tf.Variable(
                tf.truncated_normal([3, 3, 1, 32], stddev=0.1), name='W_conv11')
            b_conv11 = tf.Variable(tf.constant(0.1, shape=[32]), name='b_conv11')
            h_conv11 = tf.nn.relu(ImgConvNets._conv2d(h_trans, W_conv11) + b_conv11)

            W_conv12 = tf.Variable(
                tf.truncated_normal([3, 3, 32, 32], stddev=0.1), name='W_conv12')
            b_conv12 = tf.Variable(tf.constant(0.1, shape=[32]), name='b_conv12')
            h_conv12 = tf.nn.relu(ImgConvNets._conv2d(h_conv11, W_conv12) + b_conv12)

            # Output size: img_height/2 * img_width/2 * 32
            h_pool1 = ImgConvNets._max_pool_2x2(h_conv12)
        # Second Set of Convolutional Layers
        with tf.name_scope('conv2'):
            W_conv21 = tf.Variable(
                tf.truncated_normal([3, 3, 32, 64], stddev=0.1), name='W_conv21')
            b_conv21 = tf.Variable(tf.constant(0.1, shape=[64]), name='b_conv21')
            h_conv21 = tf.nn.relu(ImgConvNets._conv2d(h_pool1, W_conv21) + b_conv21)

            W_conv22 = tf.Variable(
                tf.truncated_normal([3, 3, 64, 64], stddev=0.1), name='W_conv22')
            b_conv22 = tf.Variable(tf.constant(0.1, shape=[64]), name='b_conv22')
            h_conv22 = tf.nn.relu(ImgConvNets._conv2d(h_conv21, W_conv22) + b_conv22)

            # Output size: img_height/4 * img_width/4 * 64
            h_pool2 = ImgConvNets._max_pool_2x2(h_conv22)
        # Fully Connected Layers
        with tf.name_scope("fully_connected"):
            para_cnt = int((self.img_height/4)*(self.img_width/4)*64)
            h_pool2_flat = tf.reshape(h_pool2, [-1, para_cnt])
            W_fc1 = tf.Variable(
                tf.truncated_normal([para_cnt, 1024], stddev=0.1), name='W_fc1')
            b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]), name='b_fc1')
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        # Dropout to reduce overfitting
        with tf.name_scope("dropout"):
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        # Readout Layer
        with tf.name_scope('readout'):
            W_fc2 = tf.Variable(
                tf.truncated_normal([1024, self.class_count], stddev=0.1), name='W_fc2')
            b_fc2 = tf.Variable(tf.constant(0.1, shape=[self.class_count]), name='b_fc2')
            logits = tf.add(tf.matmul(h_fc1_drop, W_fc2), b_fc2, name='logits')

        return logits

    def _build_inference_graph_dcnn(self, images, keep_prob):
        """
        Build initial inference graph.
        Args:
            images: Images placeholder.
            keep_prob: A placeholder for the probability that a neuron's output is kept
                    during dropout.
        Returns:
            logits: Output tensor with the computed logits.
        """
        # First Set of Convolutional Layers
        with tf.name_scope('conv1'):
            shaped_images = tf.reshape(images, [-1, self.img_height, self.img_width, 1])

            W_conv11 = tf.Variable(
                tf.truncated_normal([3, 3, 1, 32], stddev=0.1), name='W_conv11')
            b_conv11 = tf.Variable(tf.constant(0.1, shape=[32]), name='b_conv11')
            h_conv11 = tf.nn.relu(ImgConvNets._conv2d(shaped_images, W_conv11) + b_conv11)

            W_conv12 = tf.Variable(
                tf.truncated_normal([1, 3, 32, 32], stddev=0.1), name='W_conv12')
            b_conv12 = tf.Variable(tf.constant(0.1, shape=[32]), name='b_conv12')
            h_conv12 = tf.nn.relu(ImgConvNets._conv2d(h_conv11, W_conv12)+ b_conv12)

            W_conv13 = tf.Variable(
                tf.truncated_normal([3, 1, 32, 32], stddev=0.1), name='W_conv13')
            b_conv13 = tf.Variable(tf.constant(0.1, shape=[32]), name='b_conv13')
            h_conv13 = tf.nn.relu(ImgConvNets._conv2d(h_conv12, W_conv13) + b_conv13)

            W_conv14 = tf.Variable(
                tf.truncated_normal([3, 3, 32, 32], stddev=0.1), name='W_conv14')
            b_conv14 = tf.Variable(tf.constant(0.1, shape=[32]), name='b_conv14')
            h_conv14 = tf.nn.relu(ImgConvNets._conv2d(h_conv13, W_conv14) + b_conv14)

            # Output size: img_height/2 * img_width/2 * 32
            h_pool1 = ImgConvNets._max_pool_2x2(h_conv14)
        # Second Set of Convolutional Layers
        with tf.name_scope('conv2'):
            W_conv21 = tf.Variable(
                tf.truncated_normal([3, 3, 32, 64], stddev=0.1), name='W_conv21')
            b_conv21 = tf.Variable(tf.constant(0.1, shape=[64]), name='b_conv21')
            h_conv21 = tf.nn.relu(ImgConvNets._conv2d(h_pool1, W_conv21) + b_conv21)

            W_conv22 = tf.Variable(
                tf.truncated_normal([1, 3, 64, 64], stddev=0.1), name='W_conv22')
            b_conv22 = tf.Variable(tf.constant(0.1, shape=[64]), name='b_conv22')
            h_conv22 = tf.nn.relu(ImgConvNets._conv2d(h_conv21, W_conv22) + b_conv22)

            W_conv23 = tf.Variable(
                tf.truncated_normal([3, 1, 64, 64], stddev=0.1), name='W_conv23')
            b_conv23 = tf.Variable(tf.constant(0.1, shape=[64]), name='b_conv23')
            h_conv23 = tf.nn.relu(ImgConvNets._conv2d(h_conv22, W_conv23) + b_conv23)

            W_conv24 = tf.Variable(
                tf.truncated_normal([3, 3, 64, 64], stddev=0.1), name='W_conv24')
            b_conv24 = tf.Variable(tf.constant(0.1, shape=[64]), name='b_conv24')
            h_conv24 = tf.nn.relu(ImgConvNets._conv2d(h_conv23, W_conv24) + b_conv24)

            # Output size: img_height/4 * img_width/4 * 64
            h_pool2 = ImgConvNets._max_pool_2x2(h_conv24)
        # Fully Connected Layers
        with tf.name_scope("fully_connected"):
            para_cnt = int((self.img_height/4)*(self.img_width/4)*64)
            h_pool2_flat = tf.reshape(h_pool2, [-1, para_cnt])
            W_fc1 = tf.Variable(
                tf.truncated_normal([para_cnt, 1024], stddev=0.1), name='W_fc1')
            b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]), name='b_fc1')
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

            W_fc2 = tf.Variable(
                tf.truncated_normal([1024, 1024], stddev=0.1), name='W_fc2')
            b_fc2 = tf.Variable(tf.constant(0.1, shape=[1024]), name='b_fc2')
            h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
        # Dropout to reduce overfitting
        with tf.name_scope("dropout"):
            h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)
        # Readout Layer
        with tf.name_scope('readout'):
            W_fc3 = tf.Variable(
                tf.truncated_normal([1024, self.class_count], stddev=0.1), name='W_fc3')
            b_fc3 = tf.Variable(tf.constant(0.1, shape=[self.class_count]), name='b_fc3')
            logits = tf.add(tf.matmul(h_fc2_drop, W_fc3), b_fc3, name='logits')

        return logits

    def _build_inference_graph_basic(self, images, keep_prob):
        """
        Build initial inference graph.
        Args:
            images: Images placeholder.
            keep_prob: A placeholder for the probability that a neuron's output is kept
                    during dropout.
        Returns:
            logits: Output tensor with the computed logits.
        """
        # First Set of Convolutional Layers
        with tf.name_scope('conv1'):
            shaped_images = tf.reshape(images, [-1, self.img_height, self.img_width, 1])

            W_conv11 = tf.Variable(
                tf.truncated_normal([3, 3, 1, 32], stddev=0.1), name='W_conv11')
            b_conv11 = tf.Variable(tf.constant(0.1, shape=[32]), name='b_conv11')
            h_conv11 = tf.nn.relu(ImgConvNets._conv2d(shaped_images, W_conv11) + b_conv11)

            W_conv12 = tf.Variable(
                tf.truncated_normal([3, 3, 32, 32], stddev=0.1), name='W_conv12')
            b_conv12 = tf.Variable(tf.constant(0.1, shape=[32]), name='b_conv12')
            h_conv12 = tf.nn.relu(ImgConvNets._conv2d(h_conv11, W_conv12) + b_conv12)

            # Output size: img_height/2 * img_width/2 * 32
            h_pool1 = ImgConvNets._max_pool_2x2(h_conv12)
        # Second Set of Convolutional Layers
        with tf.name_scope('conv2'):
            W_conv21 = tf.Variable(
                tf.truncated_normal([3, 3, 32, 64], stddev=0.1), name='W_conv21')
            b_conv21 = tf.Variable(tf.constant(0.1, shape=[64]), name='b_conv21')
            h_conv21 = tf.nn.relu(ImgConvNets._conv2d(h_pool1, W_conv21) + b_conv21)

            W_conv22 = tf.Variable(
                tf.truncated_normal([3, 3, 64, 64], stddev=0.1), name='W_conv22')
            b_conv22 = tf.Variable(tf.constant(0.1, shape=[64]), name='b_conv22')
            h_conv22 = tf.nn.relu(ImgConvNets._conv2d(h_conv21, W_conv22) + b_conv22)

            # Output size: img_height/4 * img_width/4 * 64
            h_pool2 = ImgConvNets._max_pool_2x2(h_conv22)
        # Fully Connected Layers
        with tf.name_scope("fully_connected"):
            para_cnt = int((self.img_height/4)*(self.img_width/4)*64)
            h_pool2_flat = tf.reshape(h_pool2, [-1, para_cnt])
            W_fc1 = tf.Variable(
                tf.truncated_normal([para_cnt, 1024], stddev=0.1), name='W_fc1')
            b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]), name='b_fc1')
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        # Dropout to reduce overfitting
        with tf.name_scope("dropout"):
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        # Readout Layer
        with tf.name_scope('readout'):
            W_fc2 = tf.Variable(
                tf.truncated_normal([1024, self.class_count], stddev=0.1), name='W_fc2')
            b_fc2 = tf.Variable(tf.constant(0.1, shape=[self.class_count]), name='b_fc2')
            logits = tf.add(tf.matmul(h_fc1_drop, W_fc2), b_fc2, name='logits')

        return logits

    def _build_training_graph(self, logits, labels, learning_rate):
        """
        Build the training graph.
        Args:
            logits: Logits tensor, float - [batch_size, class_count].
            labels: Labels tensor, int32 - [batch_size], with values in the range
                [0, class_count).
            learning_rate: The learning rate for the optimization.
        Returns:
            train_op: The Op for training.
            loss: The Op for calculating loss.
        """
        with tf.variable_scope(self.model_scope):
            # Create an operation that calculates loss.
            labels = tf.to_int64(labels)
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=labels, name='xentropy')
            loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
            train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

            correct_predict = tf.nn.in_top_k(logits, labels, 1)
            accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))

        return train_op, loss, accuracy

    def _get_epoch_step_count(self, train_set_size):
        if self.max_steps > 10000:
            epoch_step = math.ceil(train_set_size / (self.batch_size * 1000.0)) * 1000
        else:
            epoch_step = math.ceil(train_set_size / (self.batch_size * 100.0)) * 100

        return epoch_step

    def _get_next_batch(self, data_set, start_index):
        cnt = data_set.shape[0]

        start = start_index
        if start >= cnt:
            start = start_index % cnt

        end = start + self.batch_size
        if end < cnt:
            return data_set[start:end]
        else:
            end = end % cnt
            return np.concatenate((data_set[start:], data_set[:end]))

    def _get_learning_rate(self, last_accu, retrain=False):
        if not self.lr_adaptive:
            return self.learning_rate
        elif last_accu >= 99.92:
            return 9.2e-5
        elif last_accu >= 99.84:
            return 1e-4
        elif last_accu >= 99.76:
            return 1.2e-4
        elif last_accu >= 99.68:
            return 1.6e-4
        elif last_accu >= 99.60:
            return 2e-4
        elif last_accu >= 99.50:
            return 2.4e-4
        elif last_accu >= 99.00:
            return 3.2e-4
        elif retrain:
            return 2e-4
        else:
            return 4e-4

    @staticmethod
    def predict(model_scope, result_dir, result_file, img_features, k=1):
        """
        Args:
            model_scope: The variable_scope used when this model was trained.
            result_dir: The full path to the folder in which the result file locates.
            result_file: The file that saves the training results.
            img_features: A 2-D ndarray (matrix) each row of which holds the pixels as
                features of one image. One or more rows (image samples) can be requested
                to be predicted at once.
            k: Optional. Number of elements to be predicted.
        Returns:
            values and indices. Refer to tf.nn.top_k for details.
        """
        with tf.Session(graph=tf.Graph()) as sess:
            saver = tf.train.import_meta_graph(os.path.join(result_dir, result_file + ".meta"))
            saver.restore(sess, os.path.join(result_dir, result_file))

            # Retrieve the Ops we 'remembered'.
            logits = tf.get_collection(model_scope+"logits")[0]
            images_placeholder = tf.get_collection(model_scope+"images")[0]
            keep_prob_placeholder = tf.get_collection(model_scope+"keep_prob")[0]

            # Add an Op that chooses the top k predictions. Apply softmax so that
            # we can have the probabilities (percentage) in the output.
            eval_op = tf.nn.top_k(tf.nn.softmax(logits), k=k)

            values, indices = sess.run(eval_op, feed_dict={images_placeholder: img_features,
                                                           keep_prob_placeholder: 1.0})

            return values, indices

    @classmethod
    def _conv2d(cls, X, W):
        return tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding='SAME')

    @classmethod
    def _max_pool_2x2(cls, X):
        return tf.nn.max_pool(X, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy import io as spio
    from skimage import color as skcolor
    from skimage import io as skio
    from skimage import transform as tsf

    from settings import PROJECT_ROOT
    from misc.imgtools import *

    training = True
    pred_type, pred_id = 0, 1

    if training:
        mat_data_file = os.path.join(PROJECT_ROOT, 'Data', 'Step4', 'Training', 'ex4data1.mat')
        data_dict = spio.loadmat(mat_data_file)
        X = data_dict['X']
        y = data_dict['y'] % 10

        l_rot_X = threshold_filtered_images(rotated_images(X, 20, 20, -25), 20, 20)
        r_rot_X = threshold_filtered_images(rotated_images(X, 20, 20, 25), 20, 20)
        new_X = np.append(np.append(X, l_rot_X, axis=0), r_rot_X, axis=0)
        new_y = np.append(np.append(y, y, axis=0), y, axis=0)

        print("X shape: {}, y shape: {}".format(new_X.shape, new_y.shape))

        res_dir = os.path.join(PROJECT_ROOT, 'Data', 'Result')
        img_cn = ImgConvNets(model='BASIC', model_scope='basic', img_height=20, img_width=20,
                             class_count=10, lr_adaptive=True, batch_size=32, max_steps=8000)
        img_cn.train(new_X, new_y, res_dir, result_file='digits_basic')
    else:
        res_dir = os.path.join(PROJECT_ROOT, 'Data', 'Result')
        img_dir = os.path.join(PROJECT_ROOT, 'Data', 'Step4', 'Test')

        if pred_type == 0:
            img_file = os.path.join(img_dir, 'digits' + str(pred_id) + '.png')
            im1 = skio.imread(img_file)
            im2 = skcolor.rgb2gray(im1)
            img_arr = tsf.resize(im2, (200, 200), order=1, mode='constant')

            images = []
            for i in range(0, 10):
                for j in range(0, 10):
                    dir_arr = img_arr[20 * i:20 * i + 20, 20 * j:20 * j + 20].T.reshape(-1)
                    images.append(dir_arr)

            X_images = np.asarray(images)
            flt_images = threshold_filtered_images(X_images, 20, 20)

            values, indices = ImgConvNets.predict('dcnn', res_dir, 'digits2_dcnn', flt_images)

            print('Predicted values for the image:')
            for i in range(0, 10):
                for j in range(0, 10):
                    print('{}\t'.format(indices[i * 10 + j][0]), end='')
                print('\n')

            plot_samples(X_images, img_height=20, img_width=20, shuffle=False)
        else:
            # Predict 10 new hand-written digits
            images = []
            for i in np.arange(0, 10):
                im1 = skio.imread(os.path.join(img_dir, str(i)+'Test.png'))
                im2 = skcolor.rgb2grey(im1)
                im3 = tsf.resize(im2, (20, 20), order=1, mode='constant')
                images.append(im3.T.reshape(-1))

            X_images = np.asarray(images)
            print("X_images shape: {}".format(X_images.shape))
            flt_images = threshold_filtered_images(X_images, 20, 20)

            values, indices = ImgConvNets.predict('dcnn', res_dir, 'digits2_dcnn', flt_images, k=3)

            for i in range(0, 10):
                print('{}th image is: {} with prob {:5.2f}% or {} with prob {:5.2f}% '
                      'or {} with prob {:5.2f}%'.
                      format(i,
                             indices[i][0], values[i][0]*100,
                             indices[i][1], values[i][1]*100,
                             indices[i][2], values[i][2]*100))

            plot_samples(X_images, img_height=20, img_width=20, shuffle=False)