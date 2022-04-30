from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow.keras import optimizers
from Inception import inception_v3
from util.util import load_datasets, load_test_dataset, load_valid_dataset
import math


def get_model():
    model = inception_v3.InceptionV3(num_class=325)

    model.build(input_shape=(None, 224, 224, 3))
    print(model.summary())

    return model


# GPU settings
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


# get the original_dataset
train_dataset = load_datasets(16)
# for i,l in train_dataset:
#     print(l)


def norm_img(image, label):
    image = tf.image.resize(image, size=(224, 224))
    return tf.cast(image, tf.float32) / 255., label

AUTOTUNE = tf.data.experimental.AUTOTUNE
train_dataset = train_dataset.map(norm_img, num_parallel_calls=AUTOTUNE)

# create model
model = get_model()

# define loss and optimizer
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
model.fit(train_dataset, batch_size=16, epochs=10)


model.save_weights('model/inseptionv3.h5')

test_dataset = load_test_dataset()
valid_dataset = load_valid_dataset()
model.predict_on_batch(test_dataset)

# loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
# optimizer = optimizers.Adam()
# #
# train_loss = tf.keras.metrics.Mean(name='train_loss')
# train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
#
# valid_loss = tf.keras.metrics.Mean(name='valid_loss')
# valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')
#
# @tf.function
# def train_step(images, labels):
#     with tf.GradientTape() as tape:
#         predictions = model.call(images, training=True)
#         loss_aux = loss_object(y_true=labels, y_pred=predictions)
#         loss = 0.5 * loss_aux + 0.5 * loss_object(y_true=labels, y_pred=predictions)
#     gradients = tape.gradient(loss, model.trainable_variables)
#     optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))
#
#     train_loss(loss)
#     train_accuracy(labels, predictions)
#
# @tf.function
# def valid_step(images, labels):
#     predictions = model.call(images, training=False)
#     v_loss = loss_object(labels, predictions)
#
#     valid_loss(v_loss)
#     valid_accuracy(labels, predictions)
#
#
# # start training
# for epoch in range(10):
#     train_loss.reset_states()
#     train_accuracy.reset_states()
#     # valid_loss.reset_states()
#     # valid_accuracy.reset_states()
#     step = 0
#     for images, labels in train_dataset:
#         step += 1
#         train_step(images, labels)
#         print("Epoch: {}/{}, step: {}/{}, loss: {:.5f},"
#               " accuracy: {:.5f}".format(epoch + 1,
#                                  10,
#                                  step,
#                                  math.ceil(47332 / 32),
#                                  train_loss.result(),
#                                  train_accuracy.result()))
#
#     for valid_images, valid_labels in valid_dataset:
#         valid_step(valid_images, valid_labels)
#
#     print("Epoch: {}/{}, train loss: {:.5f}, train accuracy: {:.5f}, "
#           .format(epoch + 1,
#                   10,
#                   train_loss.result(),
#                   train_accuracy.result()))

