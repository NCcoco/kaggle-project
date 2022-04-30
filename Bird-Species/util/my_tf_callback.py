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
from util import print_in_color


# 定制一份控制训练的代码
class LearningRateA(keras.callbacks.Callback):
    def __init__(self, base_model, model, patience, stop_patience,
                 threshold, factor, dwell, batches, initial_epoch,
                 epochs, ask_epoch):
        """
        初始化我自己的训练控制器
        :param base_model: 基础模型
        :param model: 完整模型
        :param patience: 在准确率没有提升的时候，不降低学习率的等待迭代次数
        :param stop_patience: 暂停了多少次
        :param threshold: 准确率阈值，超过阈值后，将更多的关注验证集准确率
        :param factor: 学习率降低的幅度
        :param dwell:
        :param batches: 每个训练批次大小？
        :param initial_epoch:
        :param epochs: 总训练次数
        :param ask_epoch: 保存训练结果，以在训练重新开始时恢复
        """
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


    def on_batch_begin(self, batch, logs=None):
        return super().on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs=None):
        return super().on_batch_end(batch, logs)

    def on_epoch_begin(self, epoch, logs=None):
        self.now = time.time()

    def on_epoch_end(self, epoch, logs=None):
        later = time.time()
        duration = later - self.now
        lr = float(tf.keras.backend.get_value(
            self.model.optimizer.lr))  # get the current learning rate
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
                    tf.keras.backend.set_value(self.model.optimizer.lr,
                                               lr)  # set the learning rate in the optimizer
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
                pimprov = (
                                      self.lowest_vloss - v_loss) * 100 / self.lowest_vloss
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
                    tf.keras.backend.set_value(self.model.optimizer.lr,
                                               lr)  # set the learning rate in the optimizer
                    if self.dwell:
                        self.model.set_weights(
                            self.best_weights)  # return to better point in N space
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
                    if self.base_model.trainable:
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
                        if self.base_model.trainable:
                            msg = 'base_model is already set as trainable'
                        else:
                            msg = 'setting base_model as trainable for fine tuning of model'
                            self.base_model.trainable = True
                        print_in_color(msg, (0, 255, 255), (55, 65, 80))
                        msg = '{0:^8s}{1:^10s}{2:^9s}{3:^9s}{4:^9s}{5:^9s}{6:^9s}{7:^10s}{8:^8s}'.format(
                            'Epoch',
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
                        msg = f' training will continue until epoch ' + str(
                            self.ask_epoch)
                        print_in_color(msg, (0, 255, 255), (55, 65, 80))
                        msg = '{0:^8s}{1:^10s}{2:^9s}{3:^9s}{4:^9s}{5:^9s}{6:^9s}{7:^10s}{8:10s}{9:^8s}'.format(
                            'Epoch',
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

    def on_train_batch_begin(self, batch, logs=None):
        return super().on_train_batch_begin(batch, logs)

    def on_train_batch_end(self, batch, logs=None):
        # 获取训练准确率
        acc = logs.get('accuracy') * 100
        loss = logs.get('loss')
        msg = '{0:20s}processing batch {1:4s} of {2:5s} accuracy= {3:8.3f}' \
              '  loss: {4:8.5f}'.format('', str(batch),
                                        str(self.batches),
                                        acc, loss)
        # 在同一行上打印以显示正在运行的批次
        print(msg, '\r', end='')

    def on_train_begin(self, logs=None):
        if self.base_model != None:
            status = self.base_model.trainable
            if status:
                msg = 'base_model是可训练的'
            else:
                msg = 'base_model是不可训练的'
        else:
            msg = 'base_model不存在'

        print_in_color(msg, (244, 252, 3), (55, 65, 80))
        msg = '{0:^8s}{1:^10s}{2:^9s}{3:^9s}{4:^9s}{5:^9s}{6:^9s}{7:^10s}' \
              '{8:10s}{9:^8s}'.format('Epoch', 'Loss',
                                      'Accuracy',
                                      'V_loss', 'V_acc', 'LR',
                                      'Next LR', 'Monitor',
                                      '% Improv', 'Duration')
        print_in_color(msg, (244, 252, 3), (55, 65, 80))
        self.start_time = time.time()

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
    save_id = str(model_name + '-' + subject + '-' + str(accuracy)[:str(accuracy).rfind('.') + 3] + str(time.time()) + '.h5')
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