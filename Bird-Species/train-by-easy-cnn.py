import os
import numpy as np
import pandas as pd
import tensorflow as tf
import keras.api._v2.keras as keras
from keras.api._v2.keras import layers, optimizers, losses, models,\
    regularizers
from keras.api._v2.keras.preprocessing.image import ImageDataGenerator

from util.util import *
from util.my_tf_callback import LearningRateA, saver
from sklearn.metrics import confusion_matrix, classification_report
import time

from util.datasets_util import balance, preprocessing
from util.report_util import *

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
tf.config.experimental.set_virtual_device_configuration(
    device=gpus[0],
    logical_devices=[tf.config.experimental.VirtualDeviceConfiguration(
        memory_limit=4375)]
)

# 加载数据集
dataset_name = "origin"
dataset_dir = "datasets"
categories = ["train", "test", "valid"]


# 获取所有的目录
for category in categories:
    category_dir = os.path.join(dataset_dir, category)
    # 获取该目录下的所有目录
    classes = os.listdir(category_dir)
    # 将目录下所有的文件地址放入列表中
    filenames = []
    labels = []
    for clazz in classes:
        clazz_dir = os.path.join(category_dir, clazz)
        f_names = os.listdir(clazz_dir)
        for f in f_names:
            filenames.append(os.path.join(clazz_dir, f))
            labels.append(clazz)
    # 将准备好的数据放入pandas中
    f_features = pd.Series(filenames, name="filepaths")
    labels = pd.Series(labels, name="labels")
    if category == "train":
        train_df = pd.concat([f_features, labels], axis=1)
    elif category == "test":
        test_df = pd.concat([f_features, labels], axis=1)
    else:
        valid_df = pd.concat([f_features, labels], axis=1)




# 得到pandas Dataframe
# print(train_df)
# print(test_df)
# print(valid_df)


def scalar(img):
    return img * 1./255.


# 然后将其转换为tf的数据生成器
train_gen = ImageDataGenerator(
    preprocessing_function=scalar,
    # 设置随机旋转角度 15度    # 设置随机水平翻转 # 设置随机垂直翻转
    rotation_range=15, horizontal_flip=True, vertical_flip=True)

test_valid_gen = ImageDataGenerator(preprocessing_function=scalar)

# 设置超参数
epochs = 10
batch_size = 128
learning_rate = 0.001
image_size = (224, 224)
# image_size = (150, 150)
num_classes = 325
min_sample_size = 0
max_sample_size = 140
label_column_name = "labels"
work_dir = "./datasets"
model_name = "my-cnn"


# 之前没有平衡数据集，现在采用平衡技术
# dataset_name = "balance"
# train_df, test_df, valid_df = preprocessing("datasets")
# train_df = balance(train_df, min_sample_size, max_sample_size, work_dir,
#         label_column_name, image_size)


# 获取数据加载器
train_dataloader = train_gen.flow_from_dataframe(
    # 指定图像来源列及标签列，batch_size
    train_df, x_col="filepaths", y_col="labels", batch_size=batch_size,
    # 将图像转换为的大小， 指定是分类任务
    shuffle=True, color_mode='rgb', target_size=image_size,
    class_mode="categorical")

test_dataloader = test_valid_gen.flow_from_dataframe(
    # 指定图像来源列及标签列，batch_size
    test_df, x_col="filepaths", y_col="labels", batch_size=batch_size,
    # 将图像转换为的大小， 指定是分类任务
    target_size=image_size, color_mode='rgb', class_mode="categorical"
)

valid_dataloader = test_valid_gen.flow_from_dataframe(
    # 指定图像来源列及标签列，batch_size
    valid_df, x_col="filepaths", y_col="labels", batch_size=batch_size,
    # 将图像转换为的大小， 指定是分类任务
    target_size=image_size, color_mode='rgb', class_mode="categorical"
)

# 构建模型
# 第一版训练速度太慢，而且 10 个 epochs 的准确率只能达到70%
# 两个 epochs 只达到了 27%-32% 之间
version = 1
model = models.Sequential([
    keras.Input(shape=(224, 224, 3)),
    layers.Conv2D(16, (3, 3), (1, 1), padding="valid", use_bias=True),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.Conv2D(32, (3, 3), (2, 2), padding="valid", use_bias=True),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.Conv2D(64, (3, 3), (2, 2), padding="valid", use_bias=True),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.Conv2D(128, (5, 5), (2, 2), padding="valid", use_bias=True),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.Conv2D(128, (5, 5), (2, 2), padding="valid", use_bias=True),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.AvgPool2D(pool_size=(2, 2), strides=(2, 2)),
    layers.Flatten(),
    layers.Dense(1024, activation='relu', use_bias=True),
    layers.Dense(num_classes, activation='softmax')
])


# 第二版新增2个卷积层，将图片压缩到 1x1x512
# 减少参数数量，将其中一个全连接层移除
# 第二版参数量下降45%, 但是所需内存并没有减少
# 训练速度不变，但是引入MaxPool后，模型准确率提升很大 测试集37.2%,验证集37.7%, 训练集37.2%
# version = 2
# model = models.Sequential([
#     keras.Input(shape=(150, 150, 3)),
#     layers.Conv2D(32, (3, 3), (1, 1), padding="same", use_bias=True),
#     layers.BatchNormalization(),
#     layers.ReLU(),
#     layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
#     layers.Conv2D(32, (3, 3), (1, 1), padding="valid", use_bias=True),
#     layers.BatchNormalization(),
#     layers.ReLU(),
#     layers.Conv2D(64, (3, 3), (2, 2), padding="valid", use_bias=True),
#     layers.BatchNormalization(),
#     layers.ReLU(),
#     layers.Conv2D(128, (3, 3), (2, 2), padding="valid", use_bias=True),
#     layers.BatchNormalization(),
#     layers.ReLU(),
#     layers.Conv2D(256, (3, 3), (2, 2), padding="valid", use_bias=True),
#     layers.BatchNormalization(),
#     layers.ReLU(),
#     layers.Conv2D(512, (3, 3), (1, 1), padding="valid", use_bias=True),
#     layers.BatchNormalization(),
#     layers.ReLU(),
#     layers.AvgPool2D(pool_size=(2, 2), strides=(2, 2)),
#     layers.Flatten(),
#     layers.Dense(num_classes, activation='softmax')
# ])


# 第三版，引入更多的池化层
# 比第二版多了45%的参数量，在第二版的基础上多加了4个MaxPool
# 训练速度不变，2个epochs的测试准确率49.66%，验证集准确率47.63%, 训练集47.6%
# 十次训练后，过拟合现象非常明显。测试准确率86.07%，验证集准确率46.83%, 训练集46.21%
# version = 3
# model = models.Sequential([
#     keras.Input(shape=(150, 150, 3)),
#     layers.Conv2D(32, (3, 3), (1, 1), padding="same", use_bias=True),
#     layers.BatchNormalization(),
#     layers.ReLU(),
#     layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
#     layers.Conv2D(32, (3, 3), (1, 1), padding="same", use_bias=True),
#     layers.BatchNormalization(),
#     layers.ReLU(),
#     layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
#     layers.Conv2D(64, (3, 3), (1, 1), padding="same", use_bias=True),
#     layers.BatchNormalization(),
#     layers.ReLU(),
#     layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
#     layers.Conv2D(128, (3, 3), (1, 1), padding="same", use_bias=True),
#     layers.BatchNormalization(),
#     layers.ReLU(),
#     layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
#     layers.Conv2D(256, (3, 3), (1, 1), padding="same", use_bias=True),
#     layers.BatchNormalization(),
#     layers.ReLU(),
#     layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
#     layers.Conv2D(512, (3, 3), (1, 1), padding="same", use_bias=True),
#     layers.BatchNormalization(),
#     layers.ReLU(),
#     layers.AvgPool2D(pool_size=(2, 2), strides=(2, 2)),
#     layers.Flatten(),
#     layers.Dense(num_classes, activation='softmax')
# ])

# 第四版
# 第三版的问题仍然严峻，准确率还未突破90%，并且存在极大的过拟合问题
# 再添加一层卷积加池化，将输入压缩为1 x 1 x 1024
# 尝试添加dropout，l2正则化
# 十次训练后，验证集：69.23%， 测试集：69.35%， 训练集：68.6%
# 但又产生了训练效率低的问题
# 我似乎无法想出更好的
# version = 4
# model = models.Sequential([
#     keras.Input(shape=(224, 224, 3)),
#     layers.Conv2D(64, (3, 3), (1, 1), padding="same", use_bias=True),
#     layers.BatchNormalization(),
#     layers.ReLU(),
#     layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
#     layers.Conv2D(64, (3, 3), (1, 1), padding="same", use_bias=True),
#     layers.BatchNormalization(),
#     layers.ReLU(),
#     layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
#     layers.Conv2D(128, (3, 3), (1, 1), padding="same", use_bias=True),
#     layers.BatchNormalization(),
#     layers.ReLU(),
#     layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
#     layers.Conv2D(128, (3, 3), (1, 1), padding="same", use_bias=True),
#     layers.BatchNormalization(),
#     layers.ReLU(),
#     layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
#     layers.Conv2D(256, (3, 3), (1, 1), padding="same", use_bias=True),
#     layers.BatchNormalization(),
#     layers.ReLU(),
#     layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
#     layers.Conv2D(512, (3, 3), (1, 1), padding="same", use_bias=True),
#     layers.BatchNormalization(),
#     layers.ReLU(),
#     layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
#     layers.Conv2D(1024, (3, 3), (1, 1), padding="same", use_bias=True),
#     layers.BatchNormalization(),
#     layers.ReLU(),
#     layers.AvgPool2D(pool_size=(2, 2), strides=(2, 2)),
#     layers.Flatten(),
#     layers.Dropout(rate=0.3),
#     layers.Dense(512, activation='relu',
#                  kernel_regularizer=regularizers.l2(0.01)),
#     layers.Dense(num_classes, activation='softmax')
# ])

# 试试VGG的A模型
# version = 1
# model = models.Sequential([
#     keras.Input(shape=(150, 150, 3)),
#     layers.Conv2D(64, (3, 3), (1, 1), padding="same", use_bias=True),
#     layers.BatchNormalization(),
#     layers.ReLU(),
#     layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
#     layers.Conv2D(128, (3, 3), (1, 1), padding="same", use_bias=True),
#     layers.BatchNormalization(),
#     layers.ReLU(),
#     layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
#     layers.Conv2D(256, (3, 3), (1, 1), padding="same", use_bias=True),
#     layers.BatchNormalization(),
#     layers.ReLU(),
#     layers.Conv2D(256, (3, 3), (1, 1), padding="same", use_bias=True),
#     layers.BatchNormalization(),
#     layers.ReLU(),
#     layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
#     layers.Conv2D(512, (3, 3), (1, 1), padding="same", use_bias=True),
#     layers.BatchNormalization(),
#     layers.ReLU(),
#     layers.Conv2D(512, (3, 3), (1, 1), padding="same", use_bias=True),
#     layers.BatchNormalization(),
#     layers.ReLU(),
#     layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
#     layers.Conv2D(512, (3, 3), (1, 1), padding="same", use_bias=True),
#     layers.BatchNormalization(),
#     layers.ReLU(),
#     layers.Conv2D(512, (3, 3), (1, 1), padding="same", use_bias=True),
#     layers.BatchNormalization(),
#     layers.ReLU(),
#     layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
#     layers.Flatten(),
#     # 由于原模型实在过于庞大，这里对全连接层进行优化
#     layers.Dense(1024, activation='relu'),
#     layers.Dense(1024, activation='relu'),
#     layers.Dense(num_classes, activation='softmax')
# ])

# model = models.load_model("model/cnn/vgg-A-first-68.121652849743.9300385.h5")

print(model.summary())
# exit()

model.compile(
    optimizer=optimizers.Adam(learning_rate=learning_rate),
    loss=losses.CategoricalCrossentropy(),
    metrics=['accuracy'])

history = model.fit(train_dataloader, batch_size=batch_size, epochs=epochs,
            shuffle=True, validation_data=valid_dataloader)

# acc = model.evaluate(train_dataloader, return_dict=False)[1] * 100
# print("训练集准确率为：", acc)
acc = model.evaluate(test_dataloader, return_dict=False)[1] * 100
print("测试集准确率为：", acc)
acc = model.evaluate(valid_dataloader, return_dict=False)[1] * 100
print("验证集准确率为：", acc)
print("保存路径是：", f"model/{model_name}")

save_id = f"in-{dataset_name}-{model_name}-{version}v-{epochs}epochs-" \
          f"{str(acc)[:str(acc).rfind('.') + 3]}-{time.time()}.h5"
model_save_loc = os.path.join(f"model/{model_name}", save_id)
model.save(model_save_loc)


tr_plot(history, 0)
preds = model.predict(test_dataloader)
print_info(test_dataloader, preds, 0, "tmp", model_name)


