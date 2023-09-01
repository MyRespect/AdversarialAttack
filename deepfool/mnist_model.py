import numpy as np
import tensorflow as tf

from PIL import Image
from tensorflow.data import Dataset
from tensorflow.keras import Input, Model, optimizers, datasets
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D


class MNISTModel():
    def __init__(self, shape):
        inputs = Input(shape=shape)

        x = Conv2D(32, (3, 3), activation='relu')(inputs)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Flatten()(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(32, activation='relu')(x)

        outputs = Dense(10)(x)

        self.instance = Model(inputs=inputs, outputs=outputs)


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


def loss_func(correct, predicted):
    return tf.nn.softmax_cross_entropy_with_logits(labels=correct, logits=predicted)


def train_minst(model_file='./trained/', shape=(28, 28, 1), epochs=5):
    train_ds, test_ds = load_data()
    model = MNISTModel(shape=shape).instance
    model.compile(optimizer=optimizers.Adam(learning_rate=0.003),
                  loss=loss_func, metrics=['accuracy'])
    model.fit(train_ds, epochs=epochs)
    tf.saved_model.save(model, model_file)

    results = model.evaluate(test_ds)
    print('test loss, test acc:', results)


def classifier(pic_path='testSample/img_1.jpg', model_file='./trained/', shape=(28, 28, 1)):
    image = Image.open(pic_path)
    image_array = np.array(image)
    image_norm = tf.cast(image_array / 255.0 - 0.5, tf.float32)
    image_norm = np.reshape(image_norm, shape)
    image_norm = image_norm[tf.newaxis, ...]

    model = tf.saved_model.load(model_file)
    logits = model(image_norm)

    print(logits)


train_minst()
classifier()
