import time
import numpy as np
import tensorflow as tf

from classifier import MNISTModel
from attacker import CarliniL2
from tensorflow.keras import datasets, Model, losses, optimizers, metrics


def show(img):
    """
    Show MNSIT digits in the console.
    """
    img = np.array(img)

    remap = "  .*#" + "#" * 100
    img = (img.flatten() + .5) * 3
    if len(img) != 784:
        return
    print("START")
    for i in range(28):
        print("".join([remap[int(round(x))] for x in img[i * 28:i * 28 + 28]]))


if __name__ == "__main__":

    model = MNISTModel()
    model.load_weights("./mnist/trained_model")

    attack = CarliniL2(model)

    (train_data, train_labels), (test_data,
                                 test_labels) = datasets.mnist.load_data()
    train_data, test_data = train_data / 255.0 - 0.5, test_data / 255.0 - 0.5
    train_data = train_data[..., tf.newaxis]
    test_data = test_data[..., tf.newaxis]
    train_data = tf.cast(train_data, tf.float32)
    test_data = tf.cast(test_data, tf.float32)

    # inputs = train_data[25]  # tf.Tensor[(28, 28, 1)]
    inputs = train_data[25:26]  # tf.Tensor[(1, 28, 28, 1)]

    targets = np.eye(10)[9]
    print(targets)
    list_targets = []
    list_targets.append(targets)

    timestart = time.time()
    adv = attack.attack(inputs, list_targets)
    timeend = time.time()

    print("Took", timeend - timestart, "seconds to run", len(inputs), "samples.")

    for i in range(len(adv)):
        print("Valid:")
        show(inputs[i])
        print("Classification:", model.predict(inputs[i:i + 1]))

        print("Adversarial:")
        print(np.shape(adv[i]))
        show(adv[i])

        adv[i] = adv[i][tf.newaxis, ...]
        print("Classification:", model.predict(adv[i:i + 1]))

        print("Total distortion:", np.sum((adv[i] - inputs[i])**2)**.5)
