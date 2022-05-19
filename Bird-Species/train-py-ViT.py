import os
import numpy as np
import pathlib
import pandas as pd
import keras.api._v2.keras as keras
from sklearn.metrics import confusion_matrix, classification_report
from keras.api._v2.keras import layers, \
    losses, regularizers, optimizers, applications
from keras.api._v2.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import tensorflow_hub as hub

from util.my_tf_callback import LearningRateA, saver
import util.datasets_util as ds_util
from util.util import print_in_color
import matplotlib.pyplot as plt
import math

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

# 定义一个函数创建混淆矩阵和分类报告
def print_info(test_gen, preds, print_code, save_dir, subject):
    """

    :param test_gen: 测试集数据集生成器（其指定了生成方式，通常是指向本地图片库）
    :param preds: 预测结果
    :param print_code:
    :param save_dir: 保存目录
    :param subject:
    :return:
    """
    # 获取类名及下标字典
    class_dict = test_gen.class_indices
    # 获取所有类名
    labels = test_gen.labels
    # 获取所有文件名称
    file_names = test_gen.filenames

    error_list = []
    true_class = []
    pred_class = []
    prob_list = []
    # 按下标为key 类名为value创建一个新的字典
    new_dict = {}
    error_indies = []
    # 实际预测值数组
    y_pred = []

    for key, value in class_dict.items():
        new_dict[value] = key

    # 将所有类名作为目录存储在save_dir下
    classes = list(new_dict.values())
    # 记录错误的分类次数
    errors = 0

    for i, p in enumerate(preds):
        # 预测值
        pred_index = np.argmax(p)
        # 实际值
        true_index = labels[i]
        # 如果预测错误
        if pred_index != true_index:
            error_list.append(file_names[i])
            true_class.append(new_dict[true_index])

            pred_class.append(new_dict[pred_index])
            # 预测的最高概率装进prob
            prob_list.append(p[pred_index])

            error_indies.append(true_index)
            errors = errors + 1
        y_pred.append(pred_index)
    if print_code != 0:
        if errors > 0:
            if print_code > errors:
                r = errors
            else:
                r = print_code
            msg = '{0:^28s}{1:^28s}{2:^28s}{3:^16s}' \
                .format('Filename', 'Predicted Class', 'True Class', 'Probability')
            print_in_color(msg, (0, 255, 0), (55, 65, 80))

            for i in range(r):
                # TODO 暂时不知道这几行代码干嘛的
                split1 = os.path.split(error_list[i])
                split2 = os.path.split(split1[0])
                fname = split2[1] + '/' + split1[1]

                msg = '{0:^28s}{1:^28s}{2:^28s}{3:4s}{4:^6.4f}'.format(fname, pred_class[i], true_class[i], ' ',
                                                                       prob_list[i])
                print_in_color(msg, (255, 255, 255), (55, 65, 60))

        else:
            msg = '精度为100%，没有错误'
            print_in_color(msg, (0, 255, 0), (55, 65, 80))

    if errors > 0:
        plot_bar = []
        plot_class = []
        for key, value in new_dict.items():
            # 获得被错误分类的类型的计数（例如：假设 丹顶鹤的下标是11，则下面的操作将获得实际为丹顶鹤的鸟被错误分类的数量）
            count = error_indies.count(key)
            if count != 0:
                plot_bar.append(count)
                plot_class.append(value)
        fig = plt.figure()
        fig.set_figheight(len(plot_class) / 3)
        fig.set_figwidth(10)

        for i in range(0, len(plot_class)):
            c = plot_class[i]
            x = plot_bar[i]
            plt.barh(c, x, )
            plt.title("测试集错误分类")
    y_true = np.array(labels)
    y_pred = np.array(y_pred)

    # 最多显示分类错误的30个分类
    if len(classes) <= 30:
        # 创建混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        length = len(classes)
        if length < 8:
            fig_width = 8
            fig_height = 8
        else:
            fig_width = int(length * 0.5)
            fig_height = int(length * 0.5)

        plt.figure(figsize=(fig_width, fig_height))
        plt.xticks(np.array(length) + 0.5, classes, rotation=90)
        plt.yticks(np.array(length) + 0.5, classes, rotation=0)
        plt.xlabel("预测的")
        plt.ylabel("真实的")
        plt.title("混淆矩阵")
        plt.show()

    clr = classification_report(y_true, y_pred, target_names=classes)
    print("Classification Report:\n----------------------\n", clr)


# 定义一个函数绘制训练数据
def tr_plot(tr_data, start_epoch):
    # 绘制训练数据和验证数据
    tacc = tr_data.history["accuracy"]
    tloss = tr_data.history["loss"]
    vacc = tr_data.history["val_accuracy"]
    vloss = tr_data.history["val_loss"]
    # 计算最终迭代了多少次
    Epoch_count = len(tacc) + start_epoch

    Epochs = [i + 1 for i in range(start_epoch, Epoch_count)]

    index_loss = np.argmin(vloss)
    val_lowest = vloss[index_loss]

    index_acc = np.argmax(vacc)
    acc_highest = vacc[index_acc]

    sc_label = 'best epoch=' + str(index_loss + 1 + start_epoch)
    vc_label = 'best epoch=' + str(index_acc + 1 + start_epoch)

    # 创建图表
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))
    axes[0].plot(Epochs, tloss, 'r', label='训练损失')
    axes[0].plot(Epochs, vloss, 'g', label='验证损失')
    axes[0].scatter(index_loss + 1 + start_epoch, val_lowest, s=150, c="blue", label=sc_label)
    axes[0].set_title('训练和验证损失')
    axes[0].set_xlabel("迭代次数")
    axes[0].set_ylabel("损失")
    axes[0].legend()

    axes[1].plot(Epochs, tacc, 'r', label='训练准确率')
    axes[1].plot(Epochs, vacc, 'g', label='验证准确率')
    axes[1].scatter(index_acc + 1 + start_epoch, acc_highest, s=150, c='blue', label=val_lowest)
    axes[1].set_title("训练和验证损失")
    axes[1].set_xlabel("迭代次数")
    axes[1].set_ylabel("准确率")
    axes[1].legend()


    plt.show()


# 定义一个函数，该函数对图片像素值进行压缩（0-1），
# 但由于EfficientNet网络需要0-1所以不需要进行缩放
def scalar(img):
    img = img * 1./255.
    return img


# 创建训练集、测试集、验证集
train_df, test_df, valid_df = ds_util.preprocessing("datasets")

# 设置超参数
model_name = "ViT-B_32"
ask_epoch = None
dwell = True
stop_patience = 3
patience = 1
epochs = 10
learning_rate = 0.001
factor = 0.5
dropout_p = 0.2
threshold = 0.95
freeze = True


batch_size = 128
num_classes = 325
image_size = (224, 224)
channels = 3
max_num = 140
min_num = 0
label_column_name = "labels"
work_dir = "./datasets"

test_len = len(test_df)
test_batch_size = sorted([int(test_len / n) for n in range(1, test_len + 1)
                          if test_len % n == 0 and test_len / n <= 80], reverse=True)[0]

# 平衡数据集
dataset_name = "balance"
train_df = ds_util.balance(train_df, min_num, max_num, work_dir,
                           label_column_name, image_size)


# 然后将其转换为tf的数据生成器
trgen = ImageDataGenerator(
    preprocessing_function=scalar,
    # 设置随机旋转角度 15度    # 设置随机水平翻转 # 设置随机垂直翻转
    rotation_range=15, horizontal_flip=True, vertical_flip=True)

tvgen = ImageDataGenerator(preprocessing_function=scalar)


msg = '训练集生成器'
print_in_color(msg, (0, 255, 0), (55, 65, 80))

train_gen = trgen.flow_from_dataframe(
      train_df, x_col='filepaths', y_col='labels',
      target_size=image_size, class_mode='categorical',
      color_mode='rgb', shuffle=True, batch_size=batch_size)
msg = '测试集生成器'
print_in_color(msg, (0, 255, 255), (55, 65, 80))
test_gen = tvgen.flow_from_dataframe(
     test_df, x_col='filepaths', y_col='labels',
     target_size=image_size, class_mode='categorical',
     color_mode='rgb', shuffle=False, batch_size=test_batch_size)
msg = '验证集生成器'
print_in_color(msg, (0, 255, 255), (55, 65, 80))
valid_gen = tvgen.flow_from_dataframe(
      valid_df, x_col='filepaths', y_col='labels',
      target_size=image_size, class_mode='categorical',
      color_mode='rgb', shuffle=True, batch_size=batch_size)

train_steps = int(np.ceil(len(train_gen.labels) / batch_size))
test_steps = int(test_len / test_batch_size)
valid_steps = int(np.ceil(len(valid_gen.labels) / batch_size))
batches = train_steps
# 初始化模型
version = 1
model = tf.keras.Sequential([
    layers.Input(shape=(224, 224, 3)),
    # layers.InputLayer((image_size, image_size, 3)),
    hub.KerasLayer(r"transformer/models", trainable=False),
    layers.Dropout(dropout_p),
    layers.Dense(1024, activation="relu", use_bias=True,
                 kernel_regularizer=regularizers.l2(0.02), name="fc1"),
    layers.Dense(num_classes, activation="softmax", name="fc2")
])

# 加载已初始化好的
print(model.summary())
model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate),
              loss=losses.CategoricalCrossentropy(),
              metrics=["accuracy"])


tensorboard = keras.callbacks.TensorBoard("tmp", histogram_freq=1)
callbacks = [
    LearningRateA(model=model, base_model=None, patience=patience,
                  stop_patience=stop_patience, threshold=threshold,
                factor=factor, dwell=dwell, batches=batches, initial_epoch=0,
                  epochs=epochs, ask_epoch=ask_epoch), tensorboard]
history = model.fit(x=train_gen, epochs=epochs, verbose=0,
                    callbacks=callbacks, validation_data=valid_gen,
                    validation_steps=None, shuffle=False, initial_epoch=0)
tr_plot(history, 0)


# loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
# train_loss = tf.keras.metrics.Mean(name='train_loss')
# train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
#
# optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
#
# valid_loss = tf.keras.metrics.Mean(name='valid_loss')
# valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')


# tf.config.experimental_run_functions_eagerly(True)
# @tf.function
# def train_step(images, labels, optimizer):
#     with tf.GradientTape() as tape:
#         predictions = model(images, training=True)
#         loss_aux = loss_object(y_true=labels, y_pred=predictions)
#         loss = 0.5 * loss_aux + 0.5 * loss_object(y_true=labels, y_pred=predictions)
#     gradients = tape.gradient(loss, model.trainable_variables)
#     optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))
#
#     train_loss(loss)
#     train_accuracy(labels, predictions)
#
#
# @tf.function
# def valid_step(images, labels):
#     predictions = model(images, training=False)
#     v_loss = loss_object(labels, predictions)
#
#     valid_loss(v_loss)
#     valid_accuracy(labels, predictions)
#
#
# # start training
# for epoch in range(epochs):
#     train_loss.reset_states()
#     train_accuracy.reset_states()
#     valid_loss.reset_states()
#     valid_accuracy.reset_states()
#     step = 1
#
#     while train_steps >= step:
#         images, labels = next(train_gen)
#         num_labels = []
#         for label in labels:
#             num_labels.append(np.argmax(label))
#         train_step(images, num_labels, optimizer)
#
#         print(f"Epoch: {epoch + 1}/{epochs}, "
#               f"step: {step}/{train_steps},"
#               f"learning_rate: {optimizer.lr.numpy():.7f}"
#               f" loss: {train_loss.result():.5f},"
#               f" accuracy: {train_accuracy.result():.5f}")
#         step += 1
#
#     step = 1
#     while valid_steps >= step:
#         valid_images, valid_labels = next(valid_gen)
#         num_labels = []
#         for label in valid_labels:
#             num_labels.append(np.argmax(label))
#         valid_step(valid_images, num_labels)
#         step += 1
#     print(f"Epoch: {epoch + 1}/{epochs}, "
#           f"valid loss: {valid_loss.result():.5f}, "
#           f"valid accuracy: {valid_accuracy.result():.5f}, ")
#
#     # 每训练一轮就降低80%
#     learning_rate = learning_rate * 0.6
#     keras.backend.set_value(optimizer.lr, learning_rate)


subject = 'birds'
acc = model.evaluate(test_gen, steps=test_steps, return_dict=False)[1] * 100
msg = f'accuracy on the test set is {acc:5.2f} %'
print_in_color(msg, (0, 255, 0), (55, 65, 80))
generator = train_gen
scale = 1
model_save_loc, csv_save_loc = saver(
    f"model/{model_name}", model, model_name, subject, acc, image_size, scale,
    generator, epochs=epochs, version=version, dataset_name=dataset_name)

print_code = 0
preds = model.predict(test_gen, steps=test_steps)
print_info(test_gen, preds, print_code, work_dir, subject)





