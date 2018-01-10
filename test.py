from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
import model


tf.logging.set_verbosity(tf.logging.INFO)


def main(unused_argv):
    print('Reading labels')
    with open('source/t10k-labels.idx1-ubyte', 'rb') as fd:
        eval_labels = np.asarray(model.parse_idx(fd))
    print('Reading images')
    with open('source/t10k-images.idx3-ubyte', 'rb') as fd:
        eval_data_raw = model.parse_idx(fd)

    vectors = []
    for image in eval_data_raw:
        vector = []
        for row in image:
            for value in row:
                vector.append(float(value)/255.0)
        vectors.append(vector)
    eval_data = np.asarray(vectors, dtype=np.float32)

    # exit()
    mnist_classifier = tf.estimator.Estimator(model_fn=model.cnn_model_fn, model_dir="net")

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


if __name__ == "__main__":
    tf.app.run()
