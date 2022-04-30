import tensorflow as tf
import keras.api._v2.keras as keras
from keras.api._v2.keras import backend as K
from keras.api._v2.keras.layers import Dense, Activation, Dropout, Conv2D, \
    MaxPooling2D, BatchNormalization, Flatten
from keras.api._v2.keras.optimizers import Adam, Adamax
from keras.api._v2.keras.metrics import categorical_crossentropy
from keras import regularizers
from keras.api._v2.keras.preprocessing.image import ImageDataGenerator
from keras.api._v2.keras.models import Model, load_model, Sequential
import numpy as np
import pandas as pd
# 高级文件操作库
import shutil
import time
import cv2 as cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import seaborn as sns
import os

sns.set_style('darkgrid')
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
from IPython.core.display import display, HTML
# stop annoying tensorflow warning messages
import logging


# 本工具主要用于解决繁琐的数据集处理问题


def preprocessing(str_dir):
    """
    预处理数据集，用于获取训练集、测试集、验证集
    :param str_dir: 数据集根目录
    :return:
    """
    categories = ["train", "test", "valid"]
    df = []
    for category in categories:
        # 得到具体数据集的目录
        category_path = os.path.join(str_dir, category)
        filepaths = []
        labels = []
        # 得到所有的类别（因为文件夹都是类别名称）
        classlist = os.listdir(category_path)

        for clazz in classlist:
            # 得到其中一个类别的目录
            clazz_path = os.path.join(category_path, clazz)
            f_paths = os.listdir(clazz_path)
            for f in f_paths:
                filepaths.append(os.path.join(clazz_path, f))
                labels.append(clazz)

        f_series = pd.Series(data=filepaths, name="filepaths")
        l_series = pd.Series(data=labels, name="labels")

        df.append(pd.concat([f_series, l_series], axis=1))

    #  检查每个数据集的类别数量
    for i in range(len(categories)):
        print(f"{categories[i]}有：{len(df[i])}个样本")
        print(f"{categories[i]}有：{len(df[i]['labels'].unique())}个类别")

    return df[0], df[1], df[2]


def trim(df, min_num, max_num, label_column_name):
    """
    对数据集进行裁剪
    :param df: pd的series数据集
    :param min_num: 一个类别所需要的最小样本数量
    :param max_num: 一个类别所保留的最大样本数量
    :param label_column_name:
    :return:
    """
    df = df.copy()
    classlist = list(df[label_column_name].unique())
    original_class_count = len(classlist)
    sample_list = []
    groups = df.groupby(label_column_name)

    for clazz in classlist:
        group = groups.get_group(clazz)
        count = len(group)
        if count < min_num:
            print(f"{clazz}类型的数据样本只有{count}个，不足{min_num}个已被丢弃")
        elif count > max_num:
            # 随机抛弃多余的样本
            # 这里使用sklearn提供的分割数据集函数处理
            stratify = group[label_column_name]
            train, _ = train_test_split(
                group, train_size=max_num, shuffle=True, stratify=stratify)
            sample_list.append(train)
        else:
            sample_list.append(group)

    # 过滤完之后，生成一个新的series
    df = pd.concat(sample_list, axis=0).reset_index(drop=True)
    final_class_count = len(list(df[label_column_name].unique()))
    if final_class_count != original_class_count:
        print("*** 警告： *** 类型数量减少了")
    # balance = list(df[label_column_name].value_counts())
    # print(balance)
    return df

    # 如果不太容易确定选最大值，可以通过下面的代码查看情况如何
    # groups_count = []
    # for clazz in classlist:
    #     groups_count.append(len(groups.get_group(clazz)))
    # groups_count = sorted(groups_count)
    # print(f"样本最大数量为：{max(groups_count)}")
    # print(f"样本最小值为：{min(groups_count)}")
    # print(groups_count)
    # print(np.median(groups_count))  # 通常使用中位数作为最大值
    # print(np.mean(groups_count))


def balance(train_df, min_num, max_num, work_dir, label_column_name, image_size):
    """
    平衡数据集
    :param train_df: 训练集
    :param min_num: 一个类别所需要的最小样本数量
    :param max_num: 一个类别所保留的最大样本数量
    :param work_dir: 保存增强图片的目录
    :param label_column_name: 标签列的列名称
    :return:
    """
    train_df = train_df.copy()
    train_df = trim(train_df, min_num, max_num, label_column_name)

    # 创建一个文件夹用于保存增强图片
    aug_dir = os.path.join(work_dir, "aug")
    if os.path.isdir(aug_dir):
        # 如果已经存在，则移除这个目录
        shutil.rmtree(aug_dir)
    os.mkdir(aug_dir)

    groups = train_df.groupby(label_column_name)
    classlist = list(train_df[label_column_name].unique())
    total = 0
    gen = ImageDataGenerator(
        horizontal_flip=True, rotation_range=20, zoom_range=0.2,
        channel_shift_range=15, width_shift_range=0.2, height_shift_range=0.2
    )
    for clazz in classlist:
        os.mkdir(os.path.join(aug_dir, clazz))
        group = groups.get_group(clazz)
        count = len(group)
        if count < max_num:
            aug_image_count = 0
            delta = max_num - count
            # 增强图片保存到对应类路径
            aug_class_dir = os.path.join(aug_dir, clazz)
            # 用于生成增强图片
            aug_gen = gen.flow_from_dataframe(
                group, x_col="filepaths", y_col=None, target_size=image_size,
                save_to_dir=aug_class_dir, batch_size=1, class_mode=None,
                shuffle=False, save_prefix="aug1-", color="rgb",
                save_format="jpg"
            )
            while aug_image_count < delta:
                images = next(aug_gen)
                aug_image_count += len(images)
            total += aug_image_count


    print(f"一共创建了{total}张增强图片。")
    # 将增强数据添加到dataframe中
    if total > 0:
        aug_f_paths = []
        aug_l_paths = []
        # 因为增强数据集可能与原数据集在类别上有所不同
        classlist = os.listdir(aug_dir)
        for clazz in classlist:
            aug_class_dir = os.path.join(aug_dir, clazz)
            f_paths = os.listdir(aug_class_dir)
            for f in f_paths:
                aug_f_paths.append(os.path.join(aug_class_dir, f))
                aug_l_paths.append(clazz)

        f_series = pd.Series(aug_f_paths, name="filepaths")
        l_series = pd.Series(aug_l_paths, name="labels")

        aug_df = pd.concat([f_series, l_series], axis=1)
        train_df = pd.concat([train_df, aug_df], axis=0).reset_index(drop=True)
        print(train_df.describe())
        return train_df


# a, b, c = preprocessing(r"../datasets")
# print(a.describe())
# train_df = balance(a, 0, 140, "../datasets/", "labels", (224, 224))
# print(train_df.describe())
# print(a)
# print(b)
# print(c)
