import copy
import numpy as np
import tensorflow as tf
from PIL import Image
import tensorflow.keras.backend as K
from tensorflow.keras import datasets, optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation


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
    return train_data, train_labels, test_data, test_labels


def loss_func(correct, predicted):
    return tf.nn.softmax_cross_entropy_with_logits(labels=correct, logits=predicted)


def deepfool(image, model, num_classes=10, overshoot=0.02, max_iter=50, shape=(28, 28, 1)):
    image_array = np.array(image)
    # print(np.shape(image_array)) # 28*28

    image_norm = tf.cast(image_array / 255.0 - 0.5, tf.float32)
    image_norm = np.reshape(image_norm, shape)  # 28*28*1
    image_norm = image_norm[tf.newaxis, ...]  # 1*28*28*1

    print(model(image_norm))

    f_image = model(image_norm).numpy().flatten()
    I = (np.array(f_image)).flatten().argsort()[::-1]
    I = I[0:num_classes]
    label = I[0]
    # print(label, "label")

    input_shape = np.shape(image_norm)
    pert_image = copy.deepcopy(image_norm)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0
    x = tf.Variable(pert_image)
    fs = model(x)
    k_i = label

    print(fs)  # shape=(1, 10)

    def loss_func2(labels, logits, k, I):
        # return tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        # return tf.reduce_sum(labels * tf.math.log(logits))
        return logits[0, I[k]]

    while k_i == label and loop_i < max_iter:

        pert = np.inf

        one_hot_label_0 = tf.one_hot(label, num_classes)
        with tf.GradientTape() as tape:
            tape.watch(x)
            fs = model(x)
            loss_value = loss_func2(one_hot_label_0, fs, 0, I)
            print("loss_value", loss_value)
            print("fs[0, I[0]]", fs[0, I[0]])
        # grad_orig = tape.gradient(fs[0, I[0]], x)
        grad_orig = tape.gradient(loss_value, x)

        for k in range(1, num_classes):
            one_hot_label_k = tf.one_hot(I[k], num_classes)
            with tf.GradientTape() as tape:
                tape.watch(x)
                fs = model(x)
                loss_value = loss_func2(one_hot_label_k, fs, k, I)
            # cur_grad = tape.gradient(fs[0, I[k]], x)
            cur_grad = tape.gradient(loss_value, x)

            w_k = cur_grad - grad_orig

            f_k = (fs[0, I[k]] - fs[0, I[0]]).numpy()

            pert_k = abs(f_k) / np.linalg.norm(tf.reshape(w_k, [-1]))

            if pert_k < pert:
                pert = pert_k
                w = w_k

        # print(pert)  # 1.3409956
        # print(np.shape(w))  # (1, 28, 28, 1)
        r_i = (pert + 1e-4) * w / np.linalg.norm(w)
        r_tot = np.float32(r_tot + r_i)

        pert_image = image_norm + (1 + overshoot) * r_tot

        x = tf.Variable(pert_image)

        fs = model(x)
        k_i = np.argmax(np.array(fs).flatten())

        loop_i += 1

    r_tot = (1 + overshoot) * r_tot

    return r_tot, loop_i, label, k_i, pert_image


def train_attack(model_file='./trained/', pic_path='testSample/img_2.jpg'):
    model = Sequential()
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(10))

    train_data, train_labels, test_data, test_labels = load_data()

    # model.compile(optimizer=optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy']) # loss should be softmax_cross_entropy
    model.compile(optimizer=optimizers.Adam(),
                  loss=loss_func, metrics=['accuracy'])

    # model.fit(train_data, train_labels, epochs=1, batch_size=32)

    # model.evaluate(test_data, test_labels, batch_size=32)

    # tf.saved_model.save(model, model_file)
    model = tf.saved_model.load(model_file)

    image = Image.open(pic_path)

    r, loop_i, label_orig, label_pert, pert_image = deepfool(image, model)
    print("label_orig: ", label_orig)
    print("label_pert: ", label_pert)

    print(model(pert_image))
    # print(pert_image)
    pert_image = np.reshape(pert_image, (28, 28))

    pert_image += 0.5
    pert_image *= 255
    png = Image.fromarray(pert_image.astype(np.uint8))
    png.save("./hacked.png")


train_attack()
