import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

from .util import *


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
                .format('Filename', 'Predicted Class', 'True Class',
                        'Probability')
            print_in_color(msg, (0, 255, 0), (55, 65, 80))

            for i in range(r):
                # TODO 暂时不知道这几行代码干嘛的
                split1 = os.path.split(error_list[i])
                split2 = os.path.split(split1[0])
                fname = split2[1] + '/' + split1[1]

                msg = '{0:^28s}{1:^28s}{2:^28s}{3:4s}{4:^6.4f}'.format(
                    fname, pred_class[i], true_class[i], ' ',
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