import tensorflow as tf
from tensorflow.contrib import layers

from sdcnnet import shuffle


def conv2d(x, W, b, strides=1, padding='SAME', activation=tf.nn.sigmoid):
    n = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=padding)
    n = tf.nn.bias_add(n, b)
    return activation(n)


def full(x, W, b, activation=tf.nn.sigmoid):
    n = tf.add(tf.matmul(x, W), b)
    n = activation(n)
    return n


def pool2d(x, k=2, pool_func=tf.nn.max_pool):
    return pool_func(
        x,
        ksize=[1, k, k, 1],
        strides=[1, k, k, 1],
        padding='SAME'
    )


def convVars(filter_width, filter_height,
             input_depth, output_depth,
             suffix,
             sigma=0.1, mu=0):
    w = tf.Variable(tf.truncated_normal([filter_width, filter_height,
                                         input_depth, output_depth],
                                        stddev=sigma, mean=mu),
                    name="w_" + suffix)
    b = tf.Variable(tf.zeros([output_depth]), name='b_' + suffix)
    return w, b


def flatten(net):
    return layers.flatten(net)


def fullVars(input_size, output_size,
             suffix,
             sigma=0.1, mu=0):
    w = tf.Variable(tf.truncated_normal([input_size, output_size],
                                        stddev=sigma, mean=mu),
                    name='w_' + suffix)
    b = tf.Variable(tf.zeros([output_size]), name='b_' + suffix)
    return w, b


def training_pipeline(net, one_hot_y, rate=0.001):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(net, one_hot_y)
    loss_operation = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=rate)
    training_operation = optimizer.minimize(loss_operation)
    return training_operation


class NNet:
    def __init__(self, net_factory, data, output_depth):
        super().__init__()
        self.save_path = "./tfsaves/"
        self.net_factory = net_factory
        self.data = data
        self.output_depth = output_depth

        X_train, y_train = self.data['train']
        X_validation, y_validation = self.data['validation']

        x_shape = [None]
        x_shape += X_train[0].shape
        y_shape = [None]
        y_shape += y_train[0].shape
        self.x = tf.placeholder(tf.float32, x_shape)
        self.y = tf.placeholder(tf.int32, y_shape)
        one_hot_y = tf.one_hot(self.y, self.output_depth)

        net = self.net_factory(self.x)

        self.training = training_pipeline(net, one_hot_y)

        correct_prediction = tf.equal(tf.argmax(net, 1), tf.argmax(one_hot_y, 1))
        self.accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def evaluate(self, dataset,
                 BATCH_SIZE=128):
        X_data, y_data = dataset
        num_examples = len(X_data)
        total_accuracy = 0
        sess = tf.get_default_session()
        for offset in range(0, num_examples, BATCH_SIZE):
            batch_x, batch_y = X_data[offset:offset + BATCH_SIZE], y_data[
                                                                   offset:offset + BATCH_SIZE]
            accuracy = sess.run(self.accuracy_operation,
                                feed_dict={self.x: batch_x, self.y: batch_y})
            total_accuracy += (accuracy * len(batch_x))
        return total_accuracy / num_examples

    def train(self,
              EPOCHS=10, BATCH_SIZE=128,
              save_as=None):

        dataset = self.data['train']
        X_train, y_train = dataset
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            num_examples = len(X_train)

            print("Training...")
            print()
            for i in range(EPOCHS):
                X_train, y_train = shuffle(self.data['train'])
                for offset in range(0, num_examples, BATCH_SIZE):
                    end = offset + BATCH_SIZE
                    batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                    sess.run(self.training, feed_dict={self.x: batch_x, self.y: batch_y})

                validation_accuracy = self.evaluate(self.data['validation'],
                                                    BATCH_SIZE=BATCH_SIZE)
                print("EPOCH {} ...".format(i + 1))
                print("Validation Accuracy = {:.3f}".format(validation_accuracy))
                print()

            if save_as is not None:
                saver = tf.train.Saver()
                saver.save(sess, self.save_path + save_as)
                print("Model saved as", save_as)

        print("Training done")

    def test(self, load_from=None):
        with tf.Session() as sess:
            if load_from is not None:
                print("Loading from last saved state")
                saver = tf.train.Saver()
                saver.restore(sess, tf.train.latest_checkpoint(self.save_path))

            test_accuracy = self.evaluate(self.data['test'])
            print("Test Accuracy = {:.3f}".format(test_accuracy))
