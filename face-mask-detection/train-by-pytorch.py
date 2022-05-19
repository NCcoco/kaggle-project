import os
import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# seaborn是一个基于matplotlib的可视化库
import seaborn as sns
from collections import Counter

import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

# 望文生义，将xml格式的数据转换为dict格式数据
import xmltodict

import util

# print("pytorch version:", torch.__version__)

def main():

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # 将所有文件地址放入下面的容器中
    img_paths = []
    xml_paths = []
    for dirname, _, filenames in \
            os.walk(r'D:\AI\kaggle-project\face-mask-detection\datasets'):
        for filename in filenames:
            if filename[-3:] != 'xml':
                img_paths.append(os.path.join(dirname, filename))
            else:
                xml_paths.append(os.path.join(dirname, filename))

    annotations = []
    for xml in xml_paths:
        with open(xml, 'r', encoding='utf-8') as f:
            doc = xmltodict.parse(f.read())
        objects = doc['annotation']['object']
        if type(objects) == list:
            annotations += [object1['name'] for object1 in objects]
        else:
            annotations.append(objects['name'])

    Items = Counter(annotations).keys()
    values = Counter(annotations).values()
    print(Items)
    print(values)

    # 绘制一个饼状图查数据分布情况
    background_color = '#faf9f4'
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    # 实践中，饼图添加背景颜色没有意义
    # ax1.set_facecolor(background_color)
    ax1.pie(values, wedgeprops=dict(width=0.3, edgecolor='w'),
            labels=Items, radius=1, startangle=120, autopct='%1.2f%%')
    # ax2.set_facecolor(background_color)
    ax2.bar(Items, values, color='maroon', width=0.4)
    plt.show()

    # 个人见解：
    # 虽然这个数据集极不平衡，但是只要使用时非常准确，则不用平衡数据集
    # 如果要平衡，我能想到的办法就是，将没带口罩和带错口罩数量多的图片数据作为基础，
    # 然后对其进行增强，可以得到更多这样的数据样本
    # TODO: 但有一个问题，相应的标注该怎么处理


    # 查看一下标注的图片
    # imgs_xmls = list(zip(img_paths, xml_paths))
    # random.shuffle(imgs_xmls)
    # for i in range(8):
    #     img_path, xml_path = imgs_xmls[i]
    #     util.show_rectangle_image(img_path, xml_path)
    #     plt.show()

    # 数据集创建
    options = {"with_mask": 0, "without_mask": 1, "mask_weared_incorrect": 2}
    # 定义 transform 预处理图像
    transform = transforms.Compose([
        transforms.Resize(size=(226, 226)),
        transforms.ToTensor()
    ])


    def dataset_creation(image_paths, xml_paths):
        img_tensor = []
        label_tensor = []
        for i, (img_path, xml_path) in enumerate(
                tqdm(zip(image_paths, xml_paths))):
            with open(xml_path, 'r', encoding='utf-8') as xml_f:
                doc = xmltodict.parse(xml_f.read())
            objects = doc["annotation"]["object"]
            if type(objects) == list:
                for object1 in objects:
                    x, y, w, h = list(map(int, object1["bndbox"].values()))
                    label = options[object1["name"]]
                    img = transforms.functional \
                        .crop(Image.open(img_path).convert("RGB"), y, x, h - y,
                              w - x)
                    img_tensor.append(transform(img))
                    label_tensor.append(torch.tensor(label))
            else:
                x, y, w, h = list(map(int, objects["bndbox"].values()))
                label = options[objects["name"]]
                img = transforms.functional \
                    .crop(Image.open(img_path).convert("RGB"), y, x, h - y, w - x)
                img_tensor.append(transform(img))
                label_tensor.append(torch.tensor(label))

        final_dataset = [[k, l] for k, l in zip(img_tensor, label_tensor)]
        return tuple(final_dataset)


    my_dataset = dataset_creation(img_paths, xml_paths)

    # 分割训练集与测试集
    train_size = int(len(my_dataset) * 0.7)
    test_size = len(my_dataset) - train_size
    train_ds, test_ds = torch.utils.data \
        .random_split(my_dataset, [train_size, test_size])

    print(train_size)
    print(test_size)

    train_dataloader = DataLoader(dataset=train_ds, shuffle=True,
                                  batch_size=128, num_workers=8)
    test_dataloader = DataLoader(dataset=test_ds, batch_size=128, shuffle=True,
                                 num_workers=8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_train, y_train = next(iter(train_dataloader))
    print(f"Feature batch shape: {x_train.size()}")
    print(f"Labels batch shape: {y_train.size()}")

    # 构建模型, 该模型保留了顶层
    model = models.resnet34(pretrained=True)
    print(model)
    for param in model.parameters():
        param.requires_grad = False


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=0.001)

    # 删除顶层，添加自己的顶层
    n_inputs = model.fc.in_features
    my_fc = nn.Linear(in_features=n_inputs, out_features=3)
    # TODO: 这里它为什么要加到out_features中，而不是完全替代fc?
    model.fc.out_features = my_fc
    for param in model.fc.parameters():
        param.requires_grad = True

    epochs = 5
    for epoch in range(epochs):
        running_loss = 0.0
        train_loss = []
        for i, (x, y) in enumerate(train_dataloader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 20 == 19:
                print(
                    "Epoch {}, batch {}, training loss {}"
                        .format(epoch, i + 1, running_loss / 20))
            running_loss = 0.0

    print('\nFinished Training')


if __name__ == '__main__':
    main()


# end
