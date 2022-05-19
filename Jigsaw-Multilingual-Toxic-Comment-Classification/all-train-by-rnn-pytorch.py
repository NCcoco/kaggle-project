import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score


# 实现使用所有数据进行多种类型分类
# 读取csv数据文件
train_df = pd.read_csv("datasets/jigsaw-toxic-comment-train.csv")
print(train_df.describe())
# 提取y列
train_df['y'] = (train_df[
                  ["toxic", "severe_toxic", "obscene", "threat", "insult",
                   "identity_hate"]].sum(axis=1) > 0).astype(int)
print(train_df)
# 然后从数据集中取出comment_text,y两列数据作为可训练数据集
train_df = train_df[["comment_text", "y"]].rename(
    columns={'comment_text': 'text'})
print(train_df.sample(15))

# 这个数据集是极其不平衡的
print(train_df['y'].value_counts())
print(train_df['y'].value_counts(normalize=True))

# 获取值为1的标签样本数量
min_len = (train_df['y'] == 1).sum()

# 从y==0的样本中取出与少量样本一样的样本数量
df_y0_undersample = train_df[train_df['y'] == 0].sample(n=min_len)
# 创建一个平衡的数据集
train_df = pd.concat([train_df[train_df['y'] == 1], df_y0_undersample],
                     axis=0)
print(train_df['y'].value_counts())
print(train_df['y'].value_counts(normalize=True))

# 将原本的文档集合转换为 TF-IDF 特征集
vec = TfidfVectorizer(input="content", encoding="utf-8")
X = vec.fit_transform(train_df['text'])
# 查看vec获得得词汇表
# print(vec.get_feature_names())
# 将词汇表写入到某个csv文件中
# feature_names = pd.DataFrame(data=vec.get_feature_names())
# feature_names.to_csv(path_or_buf='tmp/feature.csv', encoding='utf-8',
#                      index=False)
# print(X)
# 输出解释： (句子下标， 对应词汇表下标)， TF-IDF值
# (44935, 5866)	0.09966959874647187
# 虽然词向量编号是全矩阵统一的，但是每个词的TF-IDF值是对应该行计算的

# 训练朴素贝叶斯模型
model = MultinomialNB()
model.fit(X, train_df['y'])

# model, r = train_model(X, train_df['y'])

acc = accuracy_score(model.predict(X), train_df['y'])
print(acc)

valid_df = pd.read_csv("datasets/validation.csv")
X_toxic = vec.transform(valid_df["comment_text"])


# predict_proba 函数得出的是该数据对于各种情况的概率值
# predict 函数直接给出最大概率的类型值
p = model.predict_proba(X_toxic)
y_pred = p[:, 1]
y_pred = np.around(y_pred, 0).astype(int)
acc = accuracy_score(y_pred, valid_df['toxic'])
print(acc)
