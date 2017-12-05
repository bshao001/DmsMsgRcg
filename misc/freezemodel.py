import os
import tensorflow as tf


class FreezedModel(object):
    def __init__(self, graph, model_scope, model_dir, model_file, k=1):
        """
        Args:
            graph: The model graph.
            model_scope: The variable_scope used when this model was trained.
            model_dir: The full path to the folder in which the result file locates.
            model_file: The file that saves the training results.  
            k: Optional. Number of elements to be predicted.
        Returns:
            values and indices. Refer to tf.nn.top_k for details.
        """
        frozen_model = os.path.join(model_dir, model_file)
        with tf.gfile.GFile(frozen_model, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        tf.import_graph_def(graph_def, input_map=None, return_elements=None, name="")

        # Uncomment the two lines below to look for the names of all the operations in the graph
        # for op in graph.get_operations():
        #    print(op.name)

        # Retrieve the Ops we 'remembered'
        logits = graph.get_tensor_by_name("{}/readout/logits:0".format(model_scope))
        self.images_placeholder = graph.get_tensor_by_name("{}/images_placeholder:0".format(model_scope))
        self.keep_prob_placeholder = graph.get_tensor_by_name("{}/keep_prob_placeholder:0".format(model_scope))

        # Add an Op that chooses the top k predictions. Apply softmax so that
        # we can have the probabilities (percentage) in the output.
        self.eval_op = tf.nn.top_k(tf.nn.softmax(logits), k=k)

    def predict(self, session, img_features):
        """
        Args:
            session: The TensorFlow session.
            img_features: A 2-D ndarray (matrix) each row of which holds the pixels as
                features of one image. One or more rows (image samples) can be requested
                to be predicted at once.
        Returns:
            values and indices. Refer to tf.nn.top_k for details.
        """
        values, indices = session.run(self.eval_op, feed_dict={self.images_placeholder: img_features,
                                                               self.keep_prob_placeholder: 1.0})

        return values, indices


def freeze(model_scope, model_dir, model_file):
    """
    Args:
        model_scope: The prefix of all variables in the model.
        model_dir: The full path to the folder in which the result file locates.
        model_file: The file that saves the training results, without file suffix / extension.
    """
    saver = tf.train.import_meta_graph(os.path.join(model_dir, model_file + ".meta"))
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    with tf.Session() as sess:
        saver.restore(sess, os.path.join(model_dir, model_file))

        print("# All operations:")
        for op in graph.get_operations():
            print(op.name)

        output_node_names = [v.name.split(":")[0] for v in tf.trainable_variables()]
        output_node_names.append("{}/readout/logits".format(model_scope))
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            input_graph_def,
            output_node_names
        )

        output_file = os.path.join(model_dir, model_file + ".pb")
        with tf.gfile.GFile(output_file, "wb") as f:
            f.write(output_graph_def.SerializeToString())

        print("Freezed model was saved as {}.pb.".format(model_file))


if __name__ == "__main__":
    from settings import PROJECT_ROOT

    res_dir = os.path.join(PROJECT_ROOT, "Data", "Result")
    model_scope = "s2_lss"
    freeze(model_scope, res_dir, "s2_lss_model")
