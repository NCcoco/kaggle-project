import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import PIL.Image as Image
from sklearn.model_selection import train_test_split
import xmltodict


# 提供基本的数据查看功能
def show_images(work_dir, num=2):
    img_list = os.listdir(work_dir + "/images")
    an_dir = os.path.join(work_dir, "annotations")
    img_list, _ = train_test_split(img_list, train_size=num*num, shuffle=True)
    figs, axs = plt.subplots(num, num)
    for i, f in enumerate(img_list):
        an = f[:-4] + ".xml"
        an_path = os.path.join(work_dir, an)
        img_path = plt.imread(os.path.join(work_dir, f))
        a = int(i / num)
        b = i % num
        axs[a][b].axis('off')
        axs[a][b].imshow(img_path)


# show_images(r"D:\AI\kaggle-project\face-mask-detection\datasets", num=4)
# plt.show()


def show_rectangle_image(img_path, annotation_path):
    """
    根据标注，绘制一个完整的带有标注的图片
    :param img: 一个图片地址
    :param annotation: 一个标注文件地址
    :return:
    """
    with open(annotation_path, 'r', encoding='utf-8') as an:
        doc = xmltodict.parse(an.read())
    objects = doc['annotation']['object']
    # 准备画布
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    ax.axis('off')
    # 开始绘图
    if type(objects) == list:
        for object1 in objects:
            _get_rectangle(object1, ax)
    else:
        _get_rectangle(objects, ax)
    img = plt.imread(img_path)
    ax.imshow(img)


def _get_rectangle(object1, ax):
    """
    根据传递的object对象，决定画一个什么样的矩形
    :param object1: 单个object对象
    :return:
    """
    x, y, w, h = list(map(int, object1['bndbox'].values()))
    if object1['name'] == 'without_mask':
        mpatche = mpatches.Rectangle(
            (x, y), width=w-x, height=h-y, angle=0,
            linewidth=1, edgecolor='r', facecolor="none")
        ax.add_patch(mpatche)
        ax.annotate("without_mask", mpatche.get_xy(), color='red',
                    weight="bold", fontsize=10, ha='left', va='baseline')

    elif object1['name'] == 'with_mask':
        mpatche = mpatches.Rectangle((x, y), width=w - x, height=h - y, angle=0,
                            linewidth=1, edgecolor='g', facecolor="none")
        ax.add_patch(mpatche)
        ax.annotate("with_mask", mpatche.get_xy(), color='green',
                    weight="bold", fontsize=10, ha='left', va='baseline')
    else:
        mpatche = mpatches.Rectangle((x, y), width=w - x, height=h - y, angle=0,
                                linewidth=1, edgecolor='y', facecolor="none")
        ax.add_patch(mpatche)
        ax.annotate("mask_weared_incorrect", mpatche.get_xy(), color='yellow',
                    weight="bold", fontsize=10, ha='left', va='baseline')

