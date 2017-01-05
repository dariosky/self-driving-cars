import numpy as np


def mnist_load():
    """ get the datasets for MNIST """
    from tensorflow.examples.tutorials.mnist import input_data

    mnist = input_data.read_data_sets("data/mnist/", reshape=False)
    X_train, y_train = mnist.train.images, mnist.train.labels
    X_validation, y_validation = mnist.validation.images, mnist.validation.labels
    X_test, y_test = mnist.test.images, mnist.test.labels

    assert (len(X_train) == len(y_train))
    assert (len(X_validation) == len(y_validation))
    assert (len(X_test) == len(y_test))

    print()
    print("Image Shape: {}".format(X_train[0].shape))
    print()
    print("Training Set:   {} samples".format(len(X_train)))
    print("Validation Set: {} samples".format(len(X_validation)))
    print("Test Set:       {} samples".format(len(X_test)))
    return dict(
        train=[X_train, y_train],
        validation=[X_validation, y_validation],
        test=[X_test, y_test]
    )


def mnist_pad_to_32(data):
    # Pad images with 0s
    padding = ((0, 0), (2, 2), (2, 2), (0, 0))
    data['train'][0] = np.pad(data['train'][0], padding, 'constant')
    data['validation'][0] = np.pad(data['validation'][0], padding, 'constant')
    data['test'][0] = np.pad(data['test'][0], padding, 'constant')

    print("Updated Image Shape: {}".format(data['train'][0][0].shape))


def shuffle(dataset):
    # dataset is a list with X and Y values
    from sklearn.utils import shuffle
    return shuffle(*dataset)
