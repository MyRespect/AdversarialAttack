import numpy as np
import tensorflow as tf
from PIL import Image
from deepfool_tf import deepfool


def test_deepfool(model_file='./trained/', pic_path='testSample/img_12.jpg'):
    image = Image.open(pic_path)
    model = tf.saved_model.load(model_file)

    r, loop_i, label_orig, label_pert, pert_image = deepfool(image, model)
    print("label_orig: ", label_orig)
    print("label_pert: ", label_pert)

    pert_image = np.reshape(pert_image, (28, 28))

    pert_image += 0.5
    pert_image *= 255
    png = Image.fromarray(pert_image.astype(np.uint8))
    png.save("./hacked.png")


test_deepfool()
