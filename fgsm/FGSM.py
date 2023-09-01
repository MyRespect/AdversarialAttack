from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions


def prepare(img_path='./YellowLabradorLooking_new.jpg'):
    img = image.load_img(img_path, target_size=(299, 299))
    img = image.img_to_array(img)
    img = preprocess_input(img)
    img = img[tf.newaxis, ...]
    img = tf.Variable(img, dtype=tf.float32)

    model = InceptionV3(include_top=True, weights="imagenet")
    model.trainable = False
    return img, model


def loss_object(label, predict):
    return tf.keras.losses.categorical_crossentropy(label, predict)


def train_step(model, img, label):
    with tf.GradientTape() as tape:
        tape.watch(img)
        predict = model(img)
        # target attack, so minimize the loss
        loss = -loss_object(label, predict)
    grad = tape.gradient(loss, img)
    signed_grad = tf.sign(grad)
    # optimizer = tf.keras.optimizers.Adam()
    # optimizer.apply_gradients(zip(signed_grad, [img])) # the shape of img is auot reduced from (1,299, 299, 3) to (299, 299, 3)
    # return img
    return signed_grad


def target_attack(img_path='./YellowLabradorLooking_new.jpg', label=100, target=True, steps=100, step_alpha=1e-4):
    img, model = prepare(img_path)
    label = tf.one_hot(label, 1000)

    for i in range(steps):
        signed_grad = train_step(model, img, label)
        normed_grad = step_alpha * signed_grad
        img = img + normed_grad
        # img = train_step(model, img, label)
        if np.argmax(label) == np.argmax(model(img)):
            break
    result = model.predict(img)
    print(decode_predictions(result, top=1), i)
    return img


pert = target_attack()
pert = tf.clip_by_value(pert, 0, 1)
plt.imshow(pert[0])
plt.show()
