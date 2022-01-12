import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Activation, Dropout, Conv2D, MaxPooling2D, BatchNormalization, Flatten
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model, Sequential
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

sns.set_style('darkgrid')
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
from IPython.core.display import display, HTML
# stop annoying tensorflow warning messages
import logging

plt.rcParams['font.sans-serif'] = ['SimHei']

logging.getLogger("tensorflow").setLevel(logging.ERROR)
print('modules loaded')


# 显示数据集图片
def show_image_samples(gen):
    t_dict = gen.class_indices
    print(t_dict)
    classes = list(t_dict.keys())
    images, labels = next(gen)  # get a sample batch from the generator
    plt.figure(figsize=(20, 20))
    length = len(labels)
    if length < 25:  # show maximum of 25 images
        r = length
    else:
        r = 25
    for i in range(r):
        plt.subplot(5, 5, i + 1)
        image = images[i] / 255
        plt.imshow(image)
        index = np.argmax(labels[i])
        class_name = classes[index]
        plt.title(class_name, color='blue', fontsize=12)
        plt.axis('off')
    plt.show()


# 显示图片
def show_images(tdir):
    classlist = os.listdir(tdir)
    length = len(classlist)
    columns = 5
    rows = int(np.ceil(length / columns))
    plt.figure(figsize=(20, rows * 4))
    for i, klass in enumerate(classlist):
        classpath = os.path.join(tdir, klass)
        imgpath = os.path.join(classpath, '1.jpg')
        img = plt.imread(imgpath)
        plt.subplot(rows, columns, i + 1)
        plt.axis('off')
        plt.title(klass, color='blue', fontsize=12)
        plt.imshow(img)


# 输出颜色
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


# print_in_color("wow", (244, 252, 3), (55, 65, 80))


# 定义自定义回调的代码
class LRA(keras.callbacks.Callback):
    def __init__(self, model, base_model, patience, stop_patience, threshold, factor, dwell, batches, initial_epoch,
                 epochs, ask_epoch):
        super(LRA, self).__init__()
        self.model = model
        self.base_model = base_model
        # 指定在未调整学习率之前的遍历次数
        self.patience = patience
        # 指定调整学习率之前的训练次数后停止
        self.stop_patience = stop_patience
        # 指定当需要进行学习率调整时的训练准确率阈值
        self.threshold = threshold
        # 学习率降低的程度
        self.factor = factor
        self.dwell = dwell
        # 每个时期运行的训练批次数
        self.batches = batches
        self.initial_epoch = initial_epoch
        self.epochs = epochs
        self.ask_epoch = ask_epoch
        # 保存训练结果，以在训练重新开始时恢复
        self.ask_epoch_initial = ask_epoch
        # 回调变量  记录 lr减少了多少次的情况下，算法性能 没能得到改善
        self.count = 0
        self.stop_count = 0
        # 记录损失最低的那一次迭代的次数
        self.best_epoch = 1
        # 获取初始学习率并保存
        self.initial_lr = float(
            tf.keras.backend.get_value(model.optimizer.lr))
        # 将最高训练精度初始化为0
        self.highest_tracc = 0.0
        # 将最低验证精度设置为无穷大
        self.lowest_vloss = np.inf
        # 将最佳权重设置为初始权重（后面会进行更新）
        self.best_weights = self.model.get_weights()
        # 如果必须恢复，请保存初始权重
        self.initial_weights = self.model.get_weights()

    # 训练开始时：记录和输出一些提示信息
    def on_train_begin(self, logs=None):
        if self.base_model != None:
            status = self.base_model.trainable
            if status:
                msg = '初始化为base_model以开始训练'
            else:
                msg = '无法初始化为base_model'
        else:
            msg = '初始化模型并开始训练'

        print_in_color(msg, (244, 252, 3), (55, 65, 80))
        msg = '{0:^8s}{1:^10s}{2:^9s}{3:^9s}{4:^9s}{5:^9s}{6:^9s}{7:^10s}' \
              '{8:10s}{9:^8s}'.format('Epoch', 'Loss',
                                      'Accuracy',
                                      'V_loss', 'V_acc', 'LR',
                                      'Next LR', 'Monitor',
                                      '% Improv', 'Duration')
        print_in_color(msg, (244, 252, 3), (55, 65, 80))
        self.start_time = time.time()

    # 训练结束时：记录和输出一些提示信息
    def on_train_end(self, logs=None):
        self.stop_time = time.time()
        # 获取训练时间
        tr_duration = self.stop_time - self.start_time
        # 计算共多少小时
        hours = tr_duration // 3600
        # 计算多余时间是几分钟
        minutes = (tr_duration - (hours * 3600)) // 60
        # 计算多余秒是几秒
        seconds = tr_duration - ((hours * 3600) + (minutes * 60))
        # 设置模型的权重为之前记录的最好的权重
        self.model.set_weights(self.best_weights)
        # 训练完成，模型权重设置为最好的那次训练结果的权重
        msg = f'训练完成，模型权重设置为最好的第 {self.best_epoch} 次训练结果的权重'
        print_in_color(msg, (0, 255, 0), (55, 65, 80))
        msg = f'训练花费时间： {str(hours)} 时, {minutes:4.1f} 分, {seconds:4.2f} 秒)'
        print_in_color(msg, (0, 255, 0), (55, 65, 80))

    # 一个小批次训练结束
    def on_train_batch_end(self, batch, logs=None):
        # 获取训练准确率
        acc = logs.get('accuracy') * 100
        loss = logs.get('loss')
        msg = '{0:20s}processing batch {1:4s} of {2:5s} accuracy= {3:8.3f}  loss: {4:8.5f}'.format('', str(batch),
                                                                                                   str(self.batches),
                                                                                                   acc, loss)
        # 在同一行上打印以显示正在运行的批次
        print(msg, '\r', end='')

    # 一个训练次开始
    def on_epoch_begin(self, epoch, logs=None):
        self.now = time.time()

    # def on_epoch_end(self, epoch, logs=None):
    #     self.later = time.time()
    #     # duration 是期间的意思
    #     duration = self.later - self.now
    #     # 获取当前学习率
    #     lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
    #
    #     current_lr = lr
    #     v_loss = logs.get('val_loss')
    #     # 获取训练准确率
    #     acc = logs.get('accuracy')
    #     v_acc = logs.get('val_accuracy')
    #     loss = logs.get('loss')
    #
    #     # 如果训练精度低于阈值，则根据训练精度调整lr
    #     if acc < self.threshold:
    #         monitor = 'accuracy'
    #         if epoch == 0:
    #             pimprov = 0.0
    #         else:
    #             pimprov = (acc - self.highest_tracc) * 100 / self.highest_tracc
    #
    #         # 提高了训练精度
    #         if acc > self.highest_tracc:
    #             # 设置新的最高训练精度
    #             self.highest_tracc = acc
    #             # 将最好的权重保存到变量中
    #             self.best_weights = self.model.get_weights()
    #
    #             # 将训练精度没有提升的次数计次 归零
    #             self.count = 0
    #             self.stop_count = 0
    #
    #             if v_loss < self.lowest_vloss:
    #                 self.lowest_vloss = v_loss
    #             color = (0, 255, 0)
    #
    #             self.best_epoch = epoch + 1
    #         else:
    #             # 训练准确性没有提高检查是否超出了耐心数
    #             if self.count > self.patience - 1:
    #                 color = (245, 170, 66)
    #                 lr = lr * self.factor
    #                 # 在优化器中设置学习率
    #                 tf.keras.backend.set_value(self.model.optimizer.lr, lr)
    #                 self.count = 0
    #                 # 统计lr调整的次数
    #                 self.stop_count = self.stop_count + 1
    #
    #                 if self.dwell:
    #                     self.model.set_weights(self.best_weights)
    #                 else:
    #                     if v_loss < self.lowest_vloss:
    #                         self.lowest_vloss = v_loss
    #
    #             else:
    #                 # 增加已用耐心次数
    #                 self.count = self.count + 1
    #     # 训练准确率高于阈值，因此根据验证损失调整学习率
    #     else:
    #         # 监视
    #         monitor = 'val_loss'
    #         if epoch == 0:
    #             pimprov = 0.0
    #
    #         else:
    #             pimprov = (self.lowest_vloss - v_loss) * 100 / self.lowest_vloss
    #         # 检查验证损失是否有改进
    #         if v_loss < self.lowest_vloss:
    #             # 用新的验证损失代替旧的最低损失
    #             self.lowest_vloss = v_loss
    #             # 更换权重，该权重为最好的权重
    #             self.best_weights = self.model.get_weights()
    #             # 更换无进度统计
    #             self.count = 0
    #             self.stop_count = 0
    #             color = (0, 255, 0)
    #             # 记录这次迭代是目前为止最好的迭代
    #             self.best_epoch = epoch + 1
    #         # 损失无改进
    #         else:
    #             # 耐心耗尽，需要更新学习率
    #             if self.count >= self.patience - 1:
    #                 color = (245, 170, 66)
    #                 # 修改学习率
    #                 lr = lr * self.factor
    #                 self.stop_count = self.stop_count + 1
    #
    #                 self.count = 0
    #                 # 修改优化器中的学习率
    #                 keras.backend.set_value(self.model.optimizer.lr, lr)
    #
    #                 if self.dwell:
    #                     # 返回最好的权重
    #                     self.model.set_weights(self.best_weights)
    #             else:
    #                 # 还有耐心，继续迭代
    #                 self.count = self.count + 1
    #
    #             if acc > self.highest_tracc:
    #                 self.highest_tracc = acc
    #
    #             msg = f'{str(epoch + 1):^3s}/{str(self.epochs):4s}' \
    #                   f' {loss:^9.3f}{acc * 100:^9.3f}{v_loss:^9.5f}{v_acc * 100:^9.3f}' \
    #                   f'{current_lr:^9.5f}{lr:^9.5f}{monitor:^11s}{pimprov:^10.2f}{duration:^8.2f}'
    #
    #             print_in_color(msg, color, (55, 65, 80))
    #             if self.stop_count > self.stop_patience - 1:
    #                 # 检查学习率是否已调整 stop_count 次而没有改善学习效果
    #                 msg = f' 训练在 {epoch + 1}' \
    #                       f' 次后停止了， {self.stop_patience} 次调整学习率没有改进效果'
    #                 print_in_color(msg, (0, 255, 255), (55, 65, 80))
    #                 # 停止训练
    #                 self.model.stop_training = True
    #             else:
    #                 if self.ask_epoch != None:
    #                     if epoch + 1 >= self.ask_epoch:
    #                         if self.base_model.trainable:
    #                             msg = '输入一个H以停止训练，或者输入一个整数以继续尝试训练'
    #                         else:
    #                             msg = '输入一个H以停止训练，F微调模型，输入一个整数以继续尝试训练'
    #
    #                         print_in_color(msg, (0, 255, 255), (55, 65, 80))
    #                         ans = input('')
    #                         if ans == 'H' or ans == 'h':
    #                             msg = f'训练已在epoch {epoch + 1} 停止'
    #                             print_in_color(msg, (0, 255, 255), (55, 65, 80))
    #                             self.model.stop_training = True  # stop training
    #                         elif ans == 'F' or ans == 'f':
    #                             if self.base_model.trainable:
    #                                 msg = 'base_model 一直允许训练'
    #                             else:
    #                                 msg = '将base_model 设置为可训练以进行微调'
    #                                 self.base_model.trainable = True
    #                             print_in_color(msg, (0, 255, 255), (55, 65, 80))
    #                             msg = '{0:^8s}{1:^10s}{2:^9s}{3:^9s}{4:^9s}{5:^9s}{6:^9s}{7:^10s}{8:^8s}'.format(
    #                                 'Epoch', 'Loss', 'Accuracy',
    #                                 'V_loss', 'V_acc', 'LR', 'Next LR', 'Monitor', '% Improv', 'Duration')
    #                             print_in_color(msg, (244, 252, 3), (55, 65, 80))
    #                             self.count = 0
    #                             self.stop_count = 0
    #                             self.ask_epoch = epoch + 1 + self.ask_epoch_initial
    #
    #                         else:
    #                             ans = int(ans)
    #                             self.ask_epoch += ans
    #                             msg = f'训练将继续' + str(self.ask_epoch)
    #                             print_in_color(msg, (0, 255, 255), (55, 65, 80))
    #
    #                             msg = '{0:^8s}{1:^10s}{2:^9s}{3:^9s}{4:^9s}{5:^9s}{6:^9s}{7:^10s}{8:10s}{9:^8s}'.format(
    #                                 'Epoch', 'Loss', 'Accuracy',
    #                                 'V_loss', 'V_acc', 'LR', 'Next LR', 'Monitor', '% Improv', 'Duration')
    #                             print_in_color(msg, (244, 252, 3), (55, 65, 80))

    def on_epoch_end(self, epoch, logs=None):  # method runs on the end of each epoch
        later = time.time()
        duration = later - self.now
        lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))  # get the current learning rate
        current_lr = lr
        v_loss = logs.get('val_loss')  # get the validation loss for this epoch
        acc = logs.get('accuracy')  # get training accuracy
        v_acc = logs.get('val_accuracy')
        loss = logs.get('loss')
        if acc < self.threshold:  # if training accuracy is below threshold adjust lr based on training accuracy
            monitor = 'accuracy'
            if epoch == 0:
                pimprov = 0.0
            else:
                pimprov = (acc - self.highest_tracc) * 100 / self.highest_tracc
            if acc > self.highest_tracc:  # training accuracy improved in the epoch
                self.highest_tracc = acc  # set new highest training accuracy
                self.best_weights = self.model.get_weights()  # traing accuracy improved so save the weights
                self.count = 0  # set count to 0 since training accuracy improved
                self.stop_count = 0  # set stop counter to 0
                if v_loss < self.lowest_vloss:
                    self.lowest_vloss = v_loss
                color = (0, 255, 0)
                self.best_epoch = epoch + 1  # set the value of best epoch for this epoch
            else:
                # training accuracy did not improve check if this has happened for patience number of epochs
                # if so adjust learning rate
                if self.count >= self.patience - 1:  # lr should be adjusted
                    color = (245, 170, 66)
                    lr = lr * self.factor  # adjust the learning by factor
                    tf.keras.backend.set_value(self.model.optimizer.lr, lr)  # set the learning rate in the optimizer
                    self.count = 0  # reset the count to 0
                    self.stop_count = self.stop_count + 1  # count the number of consecutive lr adjustments
                    self.count = 0  # reset counter
                    if self.dwell:
                        self.model.set_weights(
                            self.best_weights)  # return to better point in N space
                    else:
                        if v_loss < self.lowest_vloss:
                            self.lowest_vloss = v_loss
                else:
                    self.count = self.count + 1  # increment patience counter
        else:  # training accuracy is above threshold so adjust learning rate based on validation loss
            monitor = 'val_loss'
            if epoch == 0:
                pimprov = 0.0
            else:
                pimprov = (self.lowest_vloss - v_loss) * 100 / self.lowest_vloss
            if v_loss < self.lowest_vloss:  # check if the validation loss improved
                self.lowest_vloss = v_loss  # replace lowest validation loss with new validation loss
                self.best_weights = self.model.get_weights()  # validation loss improved so save the weights
                self.count = 0  # reset count since validation loss improved
                self.stop_count = 0
                color = (0, 255, 0)
                self.best_epoch = epoch + 1  # set the value of the best epoch to this epoch
            else:  # validation loss did not improve
                if self.count >= self.patience - 1:  # need to adjust lr
                    color = (245, 170, 66)
                    lr = lr * self.factor  # adjust the learning rate
                    self.stop_count = self.stop_count + 1  # increment stop counter because lr was adjusted
                    self.count = 0  # reset counter
                    tf.keras.backend.set_value(self.model.optimizer.lr, lr)  # set the learning rate in the optimizer
                    if self.dwell:
                        self.model.set_weights(self.best_weights)  # return to better point in N space
                else:
                    self.count = self.count + 1  # increment the patience counter
                if acc > self.highest_tracc:
                    self.highest_tracc = acc
        msg = f'{str(epoch + 1):^3s}/{str(self.epochs):4s} {loss:^9.3f}{acc * 100:^9.3f}{v_loss:^9.5f}{v_acc * 100:^9.3f}{current_lr:^9.5f}{lr:^9.5f}{monitor:^11s}{pimprov:^10.2f}{duration:^8.2f}'
        print_in_color(msg, color, (55, 65, 80))
        if self.stop_count > self.stop_patience - 1:  # check if learning rate has been adjusted stop_count times with no improvement
            msg = f' training has been halted at epoch {epoch + 1} after {self.stop_patience} adjustments of learning rate with no improvement'
            print_in_color(msg, (0, 255, 255), (55, 65, 80))
            self.model.stop_training = True  # stop training
        else:
            if self.ask_epoch != None:
                if epoch + 1 >= self.ask_epoch:
                    if base_model.trainable:
                        msg = 'enter H to halt training or an integer for number of epochs to run then ask again'
                    else:
                        msg = 'enter H to halt training ,F to fine tune model, or an integer for number of epochs to run then ask again'
                    print_in_color(msg, (0, 255, 255), (55, 65, 80))
                    ans = input('')
                    if ans == 'H' or ans == 'h':
                        msg = f'training has been halted at epoch {epoch + 1} due to user input'
                        print_in_color(msg, (0, 255, 255), (55, 65, 80))
                        self.model.stop_training = True  # stop training
                    elif ans == 'F' or ans == 'f':
                        if base_model.trainable:
                            msg = 'base_model is already set as trainable'
                        else:
                            msg = 'setting base_model as trainable for fine tuning of model'
                            self.base_model.trainable = True
                        print_in_color(msg, (0, 255, 255), (55, 65, 80))
                        msg = '{0:^8s}{1:^10s}{2:^9s}{3:^9s}{4:^9s}{5:^9s}{6:^9s}{7:^10s}{8:^8s}'.format('Epoch',
                                                                                                         'Loss',
                                                                                                         'Accuracy',
                                                                                                         'V_loss',
                                                                                                         'V_acc', 'LR',
                                                                                                         'Next LR',
                                                                                                         'Monitor',
                                                                                                         '% Improv',
                                                                                                         'Duration')
                        print_in_color(msg, (244, 252, 3), (55, 65, 80))
                        self.count = 0
                        self.stop_count = 0
                        self.ask_epoch = epoch + 1 + self.ask_epoch_initial

                    else:
                        ans = int(ans)
                        self.ask_epoch += ans
                        msg = f' training will continue until epoch ' + str(self.ask_epoch)
                        print_in_color(msg, (0, 255, 255), (55, 65, 80))
                        msg = '{0:^8s}{1:^10s}{2:^9s}{3:^9s}{4:^9s}{5:^9s}{6:^9s}{7:^10s}{8:10s}{9:^8s}'.format('Epoch',
                                                                                                                'Loss',
                                                                                                                'Accuracy',
                                                                                                                'V_loss',
                                                                                                                'V_acc',
                                                                                                                'LR',
                                                                                                                'Next LR',
                                                                                                                'Monitor',
                                                                                                                '% Improv',
                                                                                                                'Duration')
                        print_in_color(msg, (244, 252, 3), (55, 65, 80))


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

    plt.tight_layout
    plt.show()


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


# 定义一个函数用于保存模型及关联csv文件
def saver(save_path, model, model_name, subject, accuracy, img_size, scalar, generator):
    """

    :param save_path: 保存路径
    :param model: 模型
    :param model_name: 模型名称
    :param subject:
    :param accuracy: 准确率
    :param img_size: 图片大小
    :param scalar:
    :param generator:
    :return:
    """
    print("保存路径是：", save_path)
    # 保存model (保存准确率的3位小数)
    save_id = str(model_name + '-' + subject + '-' + str(accuracy)[:str(accuracy).rfind('.') + 3] + '.h5')
    model_save_loc = os.path.join(save_path, save_id)
    model.save(model_save_loc)
    print_in_color('model was saved as' + model_save_loc, (0, 255, 0), (55, 65, 80))
    # 现在创建 class_df 并转换为csv文件
    class_dict = generator.class_indices
    height = []
    width = []
    scale = []
    for i in range(len(class_dict)):
        height.append(img_size[0])
        width.append(img_size[1])
        scale.append(scalar)

    # TODO 这是一个pands对象，但目前还不了解其结构
    Index_series = pd.Series(list(class_dict.values()), name='class_index')
    Class_series = pd.Series(list(class_dict.keys()), name='class')
    Height_series = pd.Series(height, name="height")
    Width_series = pd.Series(width, name='width')
    Scale_series = pd.Series(scale, name='scale by')

    class_df = pd.concat([Index_series, Class_series, Height_series, Width_series, Scale_series], axis=1)
    csv_name = 'class_dict_by_GERRY.csv'
    csv_save_loc = os.path.join(save_path, csv_name)

    class_df.to_csv(csv_save_loc, index=False)
    print_in_color('类文件已存储 ' + csv_save_loc, (0, 255, 0), (55, 65, 80))
    return model_save_loc, csv_save_loc


# 定义一个使用训练模型的函数和 csv 文件来预测图像
def predictor(sdir, csv_path, model_path, averaged=True, verbose=True):
    """

    :param sdir: 图片根目录
    :param csv_path: csv保存地址
    :param model_path: model保存地址
    :param averaged:
    :param verbose:
    :return:
    """
    # 读取 csv 文件
    class_df = pd.read_csv(csv_path)
    class_count = len(class_df['class'].unique())
    img_height = int(class_df['height'].iloc[0])
    img_width = int(class_df['width'].iloc[0])
    img_size = (img_width, img_height)

    scale = class_df['scale by'].iloc[0]

    try:
        s = int(scale)
        s2 = 1
        s1 = 0
    except:
        split = scale.split('-')
        s1 = float(split[1])
        s2 = float(split[0].split('*')[1])

    path_list = []
    paths = os.listdir(sdir)
    for f in paths:
        path_list.append(os.path.join(sdir, f))
    if verbose:
        print('10秒后加载模型')

    model = load_model(model_path)
    image_count = len(path_list)
    image_list = []
    file_list = []
    good_image_count = 0
    for i in range(image_count):
        try:
            img = cv2.imread(path_list[i])
            img = cv2.resize(img, img_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            good_image_count += 1

            # TODO 暂时没有理解
            img = img * s2 - s1
            image_list.append(img)
            file_name = os.path.split(path_list[i])[1]
            file_list.append(file_name)
        except:
            if verbose:
                print(path_list[i], 'is an invalid image file')

    # 如果只有单张图片需要扩张尺寸
    if good_image_count == 1:
        averaged = True
    image_array = np.array(image_list)
    # 对图像进行预测，对每个类别求和，然后找到最高概率的类索引
    preds = model.predict(image_array)
    if averaged:
        psum = []
        # 创造一个0数组
        for i in range(class_count):
            psum.append(0)
        # 获取其中一个预测值（是一个数组）
        for p in preds:
            for i in range(class_count):
                # TODO 这里究竟是把所有预测值放入psum，还是对概率求和？
                psum[i] = psum[i] + p[i]
        # 找到具有最高概率总和的类索引
        index = np.argmax(psum)
        # 获取该下标的类名
        klass = class_df['class'].iloc[index]
        # 获取概率的均值
        prob = psum[index] / good_image_count * 100

        for img in image_array:
            # 该方法可以修改图片的维度，原本图片的维度为(x,y),现在为（1,x,y）
            test_img = np.expand_dims(img, axis=0)
            # 找到类别概率最高的下标
            test_index = np.argmax(model.predict(test_img))

            if test_index == index:
                # 展示图片输出结果
                if verbose:
                    plt.axis('off')
                    # 显示这张图片
                    plt.imshow(img)
                    print(f'预测为{klass}，概率为{prob:6.4f}%')
                break
        return klass, prob, img, None
    # 为每个图像创建单独的预测
    else:
        pred_class = []
        prob_list = []

        for i, p in enumerate(preds):
            index = np.argmax(p)
            klass = class_df['class'].iloc[index]
            image_file = file_list[i]
            pred_class.append(klass)
            prob_list.append(p[index])

        Fseries = pd.Series(file_list, name='图片文件')
        Lseries = pd.Series(pred_class, name='？？')
        Pseries = pd.Series(prob_list, name='概率')
        df = pd.concat([Fseries, Lseries, Pseries], axis=1)
        if verbose:
            length = len(df)
            print(df.head(length))
        return None, None, None, df


# 定义一个函数，它接受一个数据帧df，整数max_size和一个字符串，
# 并返回一个数据框，其中列指定的任何类的样本数 仅限于最大样本
def trim(df, max_size, min_size, column):
    df = df.copy()
    original_class_count = len(list(df[column].unique()))
    print("数据帧中的原始类型数量：", original_class_count)
    sample_list = []
    # TODO 按我理解，这里是根据指定列进行分组，groups里面是分组好的数据
    groups = df.groupby(column)

    # 将此列数据去重然后迭代
    for label in df[column].unique():
        # 获取到该label分组下的数据
        group = groups.get_group(label)
        # 获取有多少个样本
        sample_count = len(group)
        if sample_count > max_size:
            # 如果大于最大值,则将多余数据切割丢出去
            stratify = group[column]
            samples, _ = train_test_split(group,
                                          train_size=max_size, shuffle=True,
                                          random_state=123, stratify=stratify)
            sample_list.append(samples)

        elif sample_count >= min_size:
            # 样例大于最小标准，直接使用
            sample_list.append(group)
    # 根据sample_list创建一个新的pandas对象
    df = pd.concat(sample_list, axis=0).reset_index(drop=True)
    final_class_count = len(list(df[column].unique()))
    if final_class_count != original_class_count:
        print("*** 警告： *** 类型数量减少了")
    balance = list(df[column].value_counts())
    print(balance)
    return df


# 平衡数据
def balance(train_df, max_samples, min_samples, column, working_dir, image_size):
    train_df = train_df.copy()
    train_df = trim(train_df, max_samples, min_samples, column)
    # 创建一个目录用于存放增强图片
    aug_dir = os.path.join(working_dir, 'aug')
    # 如果已经存在（通常是我们已经运行过系统所以已经生成了这些目录和文件），则移除它们
    if os.path.isdir(aug_dir):
        shutil.rmtree(aug_dir)

    os.mkdir(aug_dir)
    for label in train_df['labels'].unique():
        # 在增强文件夹下创建每个种类文件夹
        dir_path = os.path.join(aug_dir, label)
        os.mkdir(dir_path)

    # 记录 一共创建了多少增强图片
    total = 0
    # 创建并存储增强图片
    # horizontal_flip 水平翻转
    # rotation_range 旋转角度
    # zoom_range  缩放比例
    gen = ImageDataGenerator(horizontal_flip=True, rotation_range=20, width_shift_range=.2,
                             height_shift_range=.2, zoom_range=.2)

    groups = train_df.groupby('labels')
    for label in train_df['labels'].unique():
        group = groups.get_group(label)
        sample_count = len(group)
        if sample_count < max_samples:
            aug_img_count = 0
            delta = max_samples - sample_count
            target_dir = os.path.join(aug_dir, label)
            aug_gen = gen.flow_from_dataframe(group, x_col='filepaths', y_col=None, target_size=image_size,
                                              class_mode=None, batch_size=1, shuffle=False,
                                              save_to_dir=target_dir, save_prefix='aug-', color_mode='rgb',
                                              save_format='jpg')
            while aug_img_count < delta:
                images = next(aug_gen)
                aug_img_count += len(images)

            total += aug_img_count
    print(f"一共创建了增强图片{total}张")
    if total > 0:
        aug_fpaths = []
        aug_labels = []
        classlist = os.listdir(aug_dir)
        for klass in classlist:
            classpath = os.path.join(aug_dir, klass)
            flist = os.listdir(classpath)
            for f in flist:
                fpath = os.path.join(classpath, f)
                aug_fpaths.append(fpath)
                aug_labels.append(klass)
        # 将aug_fpaths 与aug_labels转换为pd对象
        Fseries = pd.Series(aug_fpaths, name='filepaths')
        Lseries = pd.Series(aug_labels, name='labels')
        # 将两个对象堆叠在一起
        aug_df = pd.concat([Fseries, Lseries], axis=1)
        # 将训练集与增强集堆叠在一起形成新的数据集
        train_df = pd.concat([train_df, aug_df], axis=0).reset_index(drop=True)

    print(list(train_df['labels'].value_counts()))
    return train_df


# 输入图像获取其shape

# img_path = r'..\datasets\train\SHOEBILL\044.jpg'
# plt.figure(figsize=(3,3))
# img = plt.imread(img_path)
# print(f"输入图片的shape={img.shape}")
# plt.axis("off")
# imshow(img)
# plt.show()

# 定义预处理函数以读取图像文件并创建数据框架
def preprocess(sdir):
    # 将3种数据集放在该数组中
    categories = ['train', 'test', 'valid']
    # 生成3个数据集，分别是训练集，测试集，验证集
    for category in categories:
        catpath = os.path.join(sdir, category)
        filepaths = []
        labels = []
        classlist = os.listdir(catpath)
        for klass in classlist:
            classpath = os.path.join(catpath, klass)
            flist = os.listdir(classpath)
            for f in flist:
                fpath = os.path.join(classpath, f)
                filepaths.append(fpath)
                labels.append(klass)

        Fseries = pd.Series(filepaths, name='filepaths')
        Lseries = pd.Series(labels, name='labels')

        if category == categories[0]:
            train_df = pd.concat([Fseries, Lseries], axis=1)
        elif category == categories[1]:
            test_df = pd.concat([Fseries, Lseries], axis=1)
        else:
            valid_df = pd.concat([Fseries, Lseries], axis=1)

    print(f"训练集长度：{len(train_df)}, 测试集长度：{len(test_df)}, 验证集长度：{len(valid_df)}")
    trcount = len(train_df['labels'].unique())
    tecount = len(test_df['labels'].unique())
    vcount = len(valid_df['labels'].unique())
    if trcount < tecount:
        msg = '** WARNING ** number of classes in training set is less than the number of classes in test set'
        print_in_color(msg, (255, 0, 0), (55, 65, 80))
        msg = 'This will throw an error in either model.evaluate or model.predict'
        print_in_color(msg, (255, 0, 0), (55, 65, 80))
    if trcount != vcount:
        msg = '** WARNING ** number of classes in training set not equal to number of classes in validation set'
        print_in_color(msg, (255, 0, 0), (55, 65, 80))
        msg = ' this will throw an error in model.fit'
        print_in_color(msg, (255, 0, 0), (55, 65, 80))
        print('train df class count: ', trcount, 'test df class count: ', tecount, ' valid df class count: ', vcount)
        ans = input('Enter C to continue execution or H to halt execution')
        if ans == 'H' or ans == 'h':
            print_in_color('Halting Execution', (255, 0, 0), (55, 65, 80))
            import sys
            sys.exit('program halted by user')
    msg = '每个类的图像计数评估数据集的平衡'
    print_in_color(msg, (0, 255, 255), (55, 65, 80))
    print(list(train_df['labels'].value_counts()))
    return train_df, test_df, valid_df


# 定义一个函数，该函数对图片像素值进行压缩（0-1），但由于EfficientNet网络需要0-255所以不需要进行缩放
def scalar(img):
    return img


sdir = r'..\datasets'
train_df, test_df, valid_df = preprocess(sdir)

# 显然，数据集是不平衡的， 设置每个种类的最大样本数来平衡数据集
max_samples = 140
min_samples = 0
column = 'labels'
working_dir = r'./'
# 压缩图片以提高训练速度 使用150 x 150 而非 224 x 224
img_size = (150, 150)
train_df = balance(train_df, max_samples, min_samples, column, working_dir, img_size)
# print(train_df)


# 创建训练、测试、验证生成器
channels = 3
batch_size = 35
img_shape = (img_size[0], img_size[1], channels)
# 获取测试集数量
length = len(test_df)

# 寻找一种能够整除的，且小于80的方案（不过我很好奇，如果length是一个质数咋整？难道用1？）
test_batch_size = sorted([int(length / n) for n in range(1, length + 1)
                          if length % n == 0 and length / n <= 80], reverse=True)[0]
test_steps = int(length / test_batch_size)
print(f"测试集每批次{test_batch_size}张图片，共分为{test_steps}批次")

trgen = ImageDataGenerator(preprocessing_function=scalar, horizontal_flip=True)
tvgen = ImageDataGenerator(preprocessing_function=scalar)

msg = '训练集生成器'
print_in_color(msg, (0, 255, 0), (55, 65, 80))
train_gen = trgen.flow_from_dataframe(train_df, x_col='filepaths', y_col='labels',
                                      target_size=img_size, class_mode='categorical',
                                      color_mode='rgb', shuffle=True, batch_size=batch_size)
msg = '测试集生成器'
print_in_color(msg, (0, 255, 255), (55, 65, 80))
test_gen = tvgen.flow_from_dataframe(test_df, x_col='filepaths', y_col='labels',
                                     target_size=img_size, class_mode='categorical',
                                     color_mode='rgb', shuffle=False, batch_size=test_batch_size)
msg = '验证集生成器'
print_in_color(msg, (0, 255, 255), (55, 65, 80))
valid_gen = tvgen.flow_from_dataframe(valid_df, x_col='filepaths', y_col='labels',
                                      target_size=img_size, class_mode='categorical',
                                      color_mode='rgb', shuffle=True, batch_size=batch_size)
classes = list(train_gen.class_indices.keys())
class_out = len(classes)
train_steps = int(np.ceil(len(train_gen.labels) / batch_size))

show_image_samples(train_gen)

# 创建模型并编译
model_name = 'EfficientNetB3'
base_model = keras.applications.EfficientNetB3(include_top=False,
                                               weights='imagenet',
                                               input_shape=img_shape, pooling='max')
# base_model = keras.models.load_model('first_EfficientNetB3-birds-99.26.h5')

x = base_model.output
x = keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
x = Dense(256, kernel_regularizer=regularizers.l2(l=0.016)
          , activity_regularizer=regularizers.l1(0.006), bias_regularizer=regularizers.l1(0.006), activation='relu')(x)
x = Dropout(rate=.45, seed=123)(x)
output = Dense(class_out, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=Adam(learning_rate=0.001), loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

# 初始化一些超参数：
# 总共迭代次数
epochs = 40
# 如果模型性能没有得到改进，最多保持学习率不变的迭代次数
patience = 1
# 如果stop_patience次更改学习率，模型性能都没能提高时，停止训练
stop_patience = 3
# 如果准确率低于此值
threshold = .95
# 修改学习率的一个指标
factor = .5

dwell = True
# 如果基础模型的初始化是由已有模型而来则为True
freeze = False
# 在停止训练前询问要运行的次数
ask_epoch = 5
batches = train_steps
callbacks = [
    LRA(model=model, base_model=base_model, patience=patience, stop_patience=stop_patience, threshold=threshold,
        factor=factor, dwell=dwell, batches=batches, initial_epoch=0, epochs=epochs, ask_epoch=ask_epoch)]
history = model.fit(x=train_gen, epochs=epochs, verbose=0, callbacks=callbacks, validation_data=valid_gen,
                    validation_steps=None, shuffle=False, initial_epoch=0)

tr_plot(history, 0)
subject = 'birds'
acc = model.evaluate(test_gen, verbose=1, steps=test_steps, return_dict=False)[1] * 100
msg = f'accuracy on the test set is {acc:5.2f} %'
print_in_color(msg, (0, 255, 0), (55, 65, 80))
generator = train_gen
scale = 1
model_save_loc, csv_save_loc = saver(working_dir, model, model_name, subject, acc, img_size, scale, generator)

print_code = 0
preds = model.predict(test_gen, steps=test_steps, verbose=1)
print_info(test_gen, preds, print_code, working_dir, subject)



# END
