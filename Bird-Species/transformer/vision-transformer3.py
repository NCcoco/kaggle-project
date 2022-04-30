import tensorflow as tf
import tensorflow_hub as hub
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import requests
import os
import platform
import pathlib
import random
import math


base_path = os.path.abspath(".")
dir_separator = "/"
if platform.system().lower() == 'windows':
    dir_separator = "\\"
    base_path = base_path[:(base_path.index('Bird-Species'))]


# 超参数设置
num_classes = 325
image_size = 224
patch_size = 32
epochs = 30
batch_size = 128
learning_rate = keras.optimizers.schedules.InverseTimeDecay(
    initial_learning_rate=0.02,
    decay_steps=100,
    decay_rate=0.7
)
learning_rate = 0.002


# 准备数据集
def load_dataset(batch_size=128):
    train_path = ['Bird-Species', 'datasets', 'train']
    # 获取所有图片地址
    train_dir = base_path + dir_separator.join(train_path)
    # 下面的方式获得一个Path类型的训练图片根路径
    train_root = pathlib.Path(train_dir)
    # # Path类型提供一个glob方法将保存的根路径下所有的文件地址分割为list
    # all_image_paths = list(train_root.glob("*/*"))
    # all_image_paths = [str(path) for path in all_image_paths]
    #
    # random.shuffle(all_image_paths)

    train_ds = keras.utils.image_dataset_from_directory(
        train_root,
        image_size=(image_size, image_size),
        batch_size=batch_size
    )
    return train_ds


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


def norm_img(image, label):
    image = tf.image.resize(image, size=(224, 224))
    return tf.cast(image, tf.float32) / 255., label


AUTOTUNE = tf.data.experimental.AUTOTUNE
train_dataset = load_dataset(batch_size)
train_dataset = train_dataset.map(norm_img, num_parallel_calls=AUTOTUNE)
train_dataset = train_dataset.cache()
train_dataset = train_dataset.prefetch(AUTOTUNE)

valid_dataset = load_valid_dataset()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()




model = tf.keras.Sequential([
    # layers.InputLayer((image_size, image_size, 3)),
    hub.KerasLayer(r"models", trainable=False),
    keras.layers.Dense(num_classes, activation="softmax")
])

model.build(input_shape=(None, 224, 224, 3))
print(model.summary())
# model.compile(optimizer='adam',
#               loss=keras.losses.SparseCategoricalCrossentropy(),
#               metrics=['accuracy'])

# model.fit(ds_train, batch_size, epochs)


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

valid_loss = tf.keras.metrics.Mean(name='valid_loss')
valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')


# tf.config.experimental_run_functions_eagerly(True)
@tf.function
def train_step(images, labels, optimizer):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss_aux = loss_object(y_true=labels, y_pred=predictions)
        loss = 0.5 * loss_aux + 0.5 * loss_object(y_true=labels, y_pred=predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


@tf.function
def valid_step(images, labels):
    predictions = model(images, training=False)
    v_loss = loss_object(labels, predictions)

    valid_loss(v_loss)
    valid_accuracy(labels, predictions)


# start training
for epoch in range(epochs):
    train_loss.reset_states()
    train_accuracy.reset_states()
    valid_loss.reset_states()
    valid_accuracy.reset_states()
    step = 0
    for images, labels in train_dataset:
        step += 1

        train_step(images, labels, optimizer)
        print(f"Epoch: {epoch + 1}/{epochs}, "
              f"step: {step}/{math.ceil(47332 / batch_size)},"
              f"learning_rate: {optimizer.lr.numpy():.7f}"
              f" loss: {train_loss.result():.5f},"
              f" accuracy: { train_accuracy.result():.5f}")

    for valid_images, valid_labels in valid_dataset:
        valid_step(valid_images, valid_labels)

    print(f"Epoch: {epoch + 1}/{epochs}, "
          f"valid loss: {valid_loss.result():.5f}, "
          f"valid accuracy: {valid_accuracy.result():.5f}, ")

    # 每训练一轮就降低80%
    learning_rate = learning_rate * 0.2
    optimizer.lr = learning_rate


# def preprocess_image(image):
#     image = np.array(image)
#     image_resized = tf.image.resize(image, (224, 224))
#     image_resized = tf.cast(image_resized, tf.float32)
#     image_resized = (image_resized - 127.5) / 127.5
#     return tf.expand_dims(image_resized, 0).numpy()
#
#
# def load_image_from_url(url):
#     response = requests.get(url)
#     image = Image.open(BytesIO(response.content))
#     image = preprocess_image(image)
#     return image
#
#
# img_url = "https://p0.pikrepo.com/preview/853/907/close-up-photo-of-gray-elephant.jpg"
# image = load_image_from_url(img_url)
# #
# # plt.imshow((image[0] + 1) / 2)
# # plt.show()
# predictions = model.predict(image)
# print(predictions)

# with open("models/ilsvrc2012_wordnet_lemmas.txt", "r") as f:
#     lines = f.readlines()
# imagenet_int_to_str = [line.rstrip() for line in lines]
#
# predicted_label = imagenet_int_to_str[int(np.argmax(predictions))]
# print(predicted_label)


