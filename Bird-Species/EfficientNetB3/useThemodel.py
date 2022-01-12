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
from LRA import preprocess, scalar, print_info


sdir = r'..\datasets'
train_df, test_df, valid_df = preprocess(sdir)

img_size = (150, 150)
channels = 3
batch_size = 35
img_shape = (img_size[0], img_size[1], channels)

length = len(test_df)
trgen = ImageDataGenerator(preprocessing_function=scalar, horizontal_flip=True)
train_gen = trgen.flow_from_dataframe(train_df, x_col='filepaths', y_col='labels',
                                      target_size=img_size, class_mode='categorical',
                                      color_mode='rgb', shuffle=True, batch_size=batch_size)
classes = list(train_gen.class_indices.keys())
scale = test_batch_size = sorted([int(length / n) for n in range(1, length + 1)
                                  if length % n == 0 and length / n <= 80], reverse=True)[0]
test_steps = int(length / test_batch_size)

tvgen = ImageDataGenerator(preprocessing_function=scalar)
test_gen = tvgen.flow_from_dataframe(test_df, x_col='filepaths', y_col='labels',
                                     target_size=img_size, class_mode='categorical',
                                     color_mode='rgb', shuffle=False, batch_size=test_batch_size)

model = keras.models.load_model('first_EfficientNetB3-birds-99.26.h5')
preds = model.predict(test_gen, steps=test_steps, verbose=1)

subject = 'birds'
working_dir = r'./'
print_code = 0
print_info(test_gen, preds, print_code, working_dir, subject)

fpath = r'../datasets/train/SHOEBILL/044.jpg'
img = plt.imread(fpath)
print('Input image shape is ', img.shape)
# resize the image so it is the same size as the images the model was trained on
img = cv2.resize(img, img_size)  # in earlier code img_size=(224,224) was used for training the model
print('the resized image has shape ', img.shape)
### show the resized image
plt.axis('off')
plt.imshow(img)
# Normally the next line of code rescales the images. However the EfficientNet model expects images in the range 0 to 255
# img= img/255
# plt.imread returns a numpy array so it is not necessary to convert the image to a numpy array
# since we have only one image we have to expand the dimensions of img so it is off the form (1,224,224,3)
# where the first dimension 1 is the batch size used by model.predict
img = np.expand_dims(img, axis=0)
print('image shape after expanding dimensions is ', img.shape)
# now predict the image
pred = model.predict(img)
print(' the shape of prediction is ', pred.shape)
# this dataset has 15 classes so model.predict will return a list of 15 probability values
# we want to find the index of the column that has the highest probability
index = np.argmax(pred[0])
# to get the actual Name of the class earlier Imade a list of the class names called classes

klass = classes[index]
# lets get the value of the highest probability
probability = pred[0][index] * 100
# print out the class, and the probability
print('the image is predicted as being ', klass, ' with a probability of ', probability)
plt.show()




