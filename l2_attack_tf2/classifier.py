import numpy as np
import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.keras import datasets, Model, losses, optimizers, metrics
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization


class MNISTModel(Model):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.conv1 = Conv2D(32, (3, 3), activation='relu',
                            input_shape=(28, 28, 1))
        self.conv2 = Conv2D(32, (3, 3), activation='relu')
        self.conv3 = Conv2D(64, (3, 3), activation='relu')
        self.conv4 = Conv2D(128, (3, 3), activation='relu')
        self.maxpo = MaxPooling2D(pool_size=(2, 2))
        self.batch = BatchNormalization()
        self.flatt = Flatten()
        self.dense = Dense(64, activation='relu')
        self.dens1 = Dense(32, activation='relu')
        self.dens2 = Dense(10)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.batch(x)
        x = self.conv3(x)
        x = self.maxpo(x)
        x = self.conv4(x)
        x = self.maxpo(x)
        x = self.flatt(x)
        x = self.dense(x)
        x = self.dens1(x)
        return self.dens2(x)


def load_data():
    (train_data, train_labels), (test_data,
                                 test_labels) = datasets.mnist.load_data()

    train_data, test_data = train_data / 255.0 - 0.5, test_data / 255.0 - 0.5
    train_data = train_data[..., tf.newaxis]
    test_data = test_data[..., tf.newaxis]
    train_data = tf.cast(train_data, tf.float32)
    test_data = tf.cast(test_data, tf.float32)

    train_labels = tf.one_hot(train_labels, 10)
    test_labels = tf.one_hot(test_labels, 10)

    train_ds = Dataset.from_tensor_slices(
        (train_data, train_labels)).shuffle(60000).batch(32)
    test_ds = Dataset.from_tensor_slices((test_data, test_labels)).batch(32)

    return train_ds, test_ds


def train(model_file='./mnist/trained_model', num_epochs=5, init=None):

    train_ds, test_ds = load_data()

    model = MNISTModel()

    optimizer = optimizers.Adam()

    train_loss = metrics.Mean(name='train_loss')
    train_accuracy = metrics.CategoricalAccuracy(name='train_accuracy')

    test_loss = metrics.Mean(name='test_loss')
    test_accuracy = metrics.CategoricalAccuracy(name='test_accuracy')

    if init != None:
        model.load_weights(model_file)

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            logits = model(images)
            loss_value = loss_object(labels, logits)
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_loss(loss_value)
        train_accuracy(labels, logits)

    @tf.function
    def test_step(images, labels):
        logits = model(images)
        loss_value = loss_object(labels, logits)
        test_loss(loss_value)
        test_accuracy(labels, logits)

    def loss_object(labels, logits):
        return tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)

    for epoch in range(num_epochs):
        for images, labels in train_ds:
            train_step(images, labels)

        for test_images, test_labels in test_ds:
            test_step(test_images, test_labels)

        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print (template.format(epoch + 1, train_loss.result(), train_accuracy.result()
                               * 100, test_loss.result(), test_accuracy.result() * 100))

    model.save_weights(model_file)


# train()
