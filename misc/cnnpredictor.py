import os
import tensorflow as tf


class CnnPredictor(object):
    """
    This predictor serves as an alternative approach to predict the classes in case we
    need to load the trained result only once, and perform prediction multiple times.
    """
    def __init__(self, session, model_scope, result_dir, result_file, k=1):
        """
        Args:
            model_scope: The variable_scope used for the trained model to be restored.
            session: The TensorFlow session used to run the prediction.
            result_dir: The full path to the folder in which the result file locates.
            result_file: The file that saves the training results.
            k: Optional. Number of elements to be predicted.
        """
        tf.train.import_meta_graph(os.path.join(result_dir, result_file + ".meta"))
        all_vars = tf.global_variables()
        model_vars = [var for var in all_vars if var.name.startswith(model_scope)]
        saver = tf.train.Saver(model_vars)
        saver.restore(session, os.path.join(result_dir, result_file))

        # Retrieve the Ops we 'remembered'.
        logits = tf.get_collection(model_scope+"logits")[0]
        self.images_placeholder = tf.get_collection(model_scope+"images")[0]
        self.keep_prob_placeholder = tf.get_collection(model_scope+"keep_prob")[0]

        # Add an Op that chooses the top k predictions. Apply softmax so that
        # we can have the probabilities (percentage) in the output.
        self.eval_op = tf.nn.top_k(tf.nn.softmax(logits), k=k)
        self.session = session

    def predict(self, img_features):
        """
        Args:
            img_features: A 2-D ndarray (matrix) each row of which holds the pixels as
                features of one image. One or more rows (image samples) can be requested
                to be predicted at once.
        Returns:
            values and indices. Refer to tf.nn.top_k for details.
        """
        pred = self.session.run(self.eval_op,
                                feed_dict={self.images_placeholder: img_features,
                                           self.keep_prob_placeholder: 1.0})
        return pred.values, pred.indices
