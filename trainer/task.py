import numpy as np
import tensorflow as tf
from keras.optimizers import SGD
import argparse
from tensorflow.python.lib.io import file_io



from model import get_model
from util import get_data, plot_multi_class_roc


def main(job_dir, **args):
    randomSeed = 86754231
    np.random.seed(randomSeed)

    # Tensorflow directory
    logs_path = job_dir + 'logs/tensorboard_logs'

    with tf.device('/device:GPU:0'):
        (x_train, y_train), (x_test, y_test) = get_data()

        print("Train Samples", x_train.shape[0])
        print("Test Samples", x_test.shape[0])
        print("Train Shape", x_train.shape, y_train.shape)

        optimizer = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
        loss = 'categorical_crossentropy'
        metrics = ['accuracy']

        model = get_model()
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

        model.fit(x_train, y_train, nb_epoch=10, batch_size=256, verbose=1, validation_data=(x_test, y_test))

        y_score = model.predict(x_test)
        plot_multi_class_roc(y_test, y_score)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Input Arguments
    parser.add_argument(
        '--job-dir',
        help='GCS location to write checkpoints and export models',
        required=True
    )
    args = parser.parse_args()
    arguments = args.__dict__

    main(**arguments)
