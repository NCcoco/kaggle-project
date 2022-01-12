import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

def simple_conv():
    """
    经过测试，此最简单的一个卷积神经网络，在10次训练的情况下，就可以将准确率提升至89%
    :return:
    """
    model = keras.models.Sequential([
        # 输出 224-5/2 + 1 = 110
        keras.layers.Conv2D(filters=32, kernel_size=5, strides=2, padding='valid', activation='relu'),
        # 输出 55
        keras.layers.MaxPooling2D(pool_size=2, padding='valid'),
        keras.layers.Flatten(),
        keras.layers.Dense(325)
    ])
    return model
