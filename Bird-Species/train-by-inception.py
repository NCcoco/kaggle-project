import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.api._v2.keras import layers, losses, optimizers, models,\
    applications
from keras.api._v2.keras.preprocessing.image import ImageDataGenerator
from util.datasets_util import preprocessing, balance

from util.report_util import *


gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
tf.config.experimental.set_virtual_device_configuration(
    device=gpus[0],
    logical_devices=[tf.config.experimental.VirtualDeviceConfiguration(
        memory_limit=6000)]
)

# 设置超参数
epochs = 10
batch_size = 1
learning_rate = 0.001
# image_size = (224, 224)
image_size = (150, 150)
num_classes = 325
min_sample_size = 0
max_sample_size = 140
label_column_name = "labels"
work_dir = "./datasets"
model_name = "inception-v3"


# 获取数据
def load_data():

    dataset_name = "balance"
    train_df, test_df, valid_df = preprocessing("datasets")
    train_df = balance(train_df, min_sample_size, max_sample_size, work_dir,
                       label_column_name, image_size)

    # dataset_name = "origin"
    # train_df, test_df, valid_df = preprocessing("datasets")
    return dataset_name, train_df, test_df, valid_df


dataset_name, train_df, test_df, valid_df = load_data()


def scalar(img):
    return img * 1./255.


train_gen = ImageDataGenerator(
    preprocessing_function=scalar,
    # 设置随机旋转角度 15度    # 设置随机水平翻转 # 设置随机垂直翻转
    rotation_range=15, horizontal_flip=True, vertical_flip=True)

test_valid_gen = ImageDataGenerator(preprocessing_function=scalar)

# 创建数据加载器
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

# models
version = 2
base_model = applications.inception_v3.InceptionV3(
    include_top=False, weights=None, input_shape=(150, 150, 3)
)
x = base_model.output
x = layers.Flatten()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(1024, activation='relu', use_bias=True)(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(num_classes, activation='softmax')(x)
model = models.Model(inputs=base_model.input, outputs=x)
print(model.summary())
model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate),
              loss=losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

history = model.fit(train_dataloader, batch_size=batch_size, epochs=epochs,
                    validation_data=valid_dataloader)

# 测试集及验证集
acc = model.evaluate(test_dataloader, return_dict=False)[1] * 100
print("测试集准确率：", acc)
acc = model.evaluate(valid_dataloader, return_dict=False)[1] * 100
print("验证集准确率：", acc)

print("保存路径是：", f"model/{model_name}")

save_id = f"in-{dataset_name}-{model_name}-{version}v-{epochs}epochs-" \
          f"{str(acc)[:str(acc).rfind('.') + 3]}-{time.time()}.h5"
model_save_loc = os.path.join(f"model/{model_name}", save_id)
model.save(model_save_loc)

#
tr_plot(history, 0)
preds = model.predict(test_dataloader)
print_info(test_dataloader, preds, 0, "tmp", model_name)

