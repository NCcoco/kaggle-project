# 引入必要的库
import functools

import imageio
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
import matplotlib.pyplot as plt
import os
import pathlib
import platform
import random
from PIL import Image

# 区分mac和window
# 同时用两个系统真是麻烦
base_path = os.path.abspath(".")
dir_separator = "/"
if platform.system().lower() == 'windows':
    dir_separator = "\\"
    base_path = base_path[:(base_path.index('Bird-Species'))]

AUTOTUNE = tf.data.experimental.AUTOTUNE


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (224, 224))
    image = tf.cast(image, dtype='float32')
    image /= 255.0

    return image


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)


def load_and_preprocess_from_path_label(path, label):
    return load_and_preprocess_image(path), label


# 加载鸟类图片数据和标签
def load_datasets(batch_size=124):
    train_path = ['Bird-Species', 'datasets', 'train']
    train_dir = base_path + dir_separator.join(train_path)
    # return __load_dataset(train_dir, batch_size=batch_size, image_size=(224,224))
    data_root = pathlib.Path(train_dir)
    # 获取所有的图片路径
    all_image_paths = list(data_root.glob('*/*'))
    all_image_paths = [str(path) for path in all_image_paths]
    # 打乱路径list
    random.shuffle(all_image_paths)
    image_count = len(all_image_paths)
    # print(all_image_paths[:10])

    # c = np.array(imageio.imread(all_image_paths[0]))
    # plt.imshow(c)
    # plt.show()

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_root,
        image_size=(224, 224),
        batch_size=batch_size)
    # print(train_ds)
    class_names = train_ds.class_names
    # print(class_names)
    # plt.figure(figsize=(10, 10))
    # for images, labels in train_ds.take(1):
    #     for i in range(9):
    #         ax = plt.subplot(3, 3, i + 1)
    #         plt.imshow(images[i].numpy().astype("uint8"))
    #         plt.title(class_names[labels[i]])
    #         plt.axis("off")
    # plt.show()

    # normalization_layer = tf.keras.layers.Rescaling(1. / 255)
    # normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

    # train_ds = normalized_ds.cache().prefetch(buffer_size=AUTOTUNE)
    return train_ds


# 加载测试集数据
def load_test_dataset():
    test_dir = ['Bird-Species', 'datasets', 'test']
    test_dir = base_path + dir_separator.join(test_dir)
    return __load_dataset(test_dir, batch_size=124, image_size=(224, 224))


# 加载验证集
def load_valid_dataset():
    valid_dir = ['Bird-Species', 'datasets', 'valid']
    valid_dir = base_path + dir_separator.join(valid_dir)
    return __load_dataset(valid_dir)


def __load_dataset(dir, batch_size=64, image_size=(224, 224)):
    data_root = pathlib.Path(dir)
    # 获取所有的图片路径
    all_image_paths = list(data_root.glob('*/*'))
    all_image_paths = [str(path) for path in all_image_paths]
    # 打乱路径list
    random.shuffle(all_image_paths)
    image_count = len(all_image_paths)
    # print(all_image_paths[:10])

    # c = np.array(imageio.imread(all_image_paths[0]))
    # plt.imshow(c)
    # plt.show()

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_root,
        image_size=image_size,
        batch_size=batch_size)
    # print(train_ds)
    class_names = train_ds.class_names
    # print(class_names)
    # plt.figure(figsize=(10, 10))
    # for images, labels in train_ds.take(1):
    #     for i in range(9):
    #         ax = plt.subplot(3, 3, i + 1)
    #         plt.imshow(images[i].numpy().astype("uint8"))
    #         plt.title(class_names[labels[i]])
    #         plt.axis("off")
    # plt.show()

    normalization_layer = tf.keras.layers.Rescaling(1. / 255)
    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

    # train_ds = normalized_ds.cache().prefetch(buffer_size=AUTOTUNE)
    return normalized_ds



def print_in_color(txt_msg, fore_tupple, back_tupple):
    # prints the text_msg in the foreground color specified by fore_tupple with the background specified by back_tupple
    # text_msg is the text, fore_tupple is foregroud color tupple (r,g,b), back_tupple is background tupple (r,g,b)
    rf, gf, bf = fore_tupple
    rb, gb, bb = back_tupple
    msg = '{0}' + txt_msg
    mat = '\33[38;2;' + str(rf) + ';' + str(gf) + ';' + str(bf) + ';48;2;' + str(rb) + ';' + str(gb) + ';' + str(
        bb) + 'm'
    print(msg.format(mat), flush=True)
    print('\33[0m', flush=True)  # returns default print color to back to black
    return

# 得到数据
# train_ds = load_datasets()

# ds = train_ds.batch(64)
# print(train_ds)
# h = inception_net(ds)

# model.fit(X_train, Y_train, epochs=10, batch_size=32)
# preds = model.evaluate(X_test, Y_test)
# model.save('Resnet/ResNet50.h5')
# print("Loss = " + str(preds[0]))
# print("Test Accuracy = " + str(preds[1]))

# model = simple_conv()
# model.compile(optimizer='adam',
#               loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])
# model.fit(train_ds, batch_size=64, epochs=10)
# test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
# print('\nTest accuracy:', test_acc)
