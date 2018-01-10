from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
import model


tf.logging.set_verbosity(tf.logging.INFO)


def main(unused_argv):
    # Load training and eval data
    print('Reading labels')
    with open('source/train-labels.idx1-ubyte', 'rb') as fd:
        train_labels = np.asarray(model.parse_idx(fd))
    print('Reading images')
    with open('source/train-images.idx3-ubyte', 'rb') as fd:
        train_data_raw = model.parse_idx(fd)

    vectors = []
    for image in train_data_raw:
        vector = []
        for row in image:
            for value in row:
                vector.append(float(value)/255.0)
        vectors.append(vector)
    train_data = np.asarray(vectors, dtype=np.float32)

    mnist_classifier = tf.estimator.Estimator(model_fn=model.cnn_model_fn, model_dir="net")
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log,
        every_n_iter=50
    )
    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True
    )
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=1000,
        hooks=[logging_hook]
    )


if __name__ == "__main__":
    tf.app.run()
