import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.api._v2 import keras
from keras.api._v2.keras.layers import Dense, RNN, GRU, LSTM, SimpleRNN, \
    BatchNormalization, Embedding
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
pd.set_option('display.max_columns', 8)
pd.set_option('display.max_rows', 15)
pd.set_option('display.width', 100)


# 使用GPU
strategy = tf.distribute.get_strategy()

# 获取数据集
train_df = pd.read_csv("datasets/jigsaw-toxic-comment-train.csv")
valid_df = pd.read_csv("datasets/validation.csv")
test_df = pd.read_csv("datasets/test.csv")

print(train_df.describe())
print(test_df.describe())
print(valid_df.describe())

# 移除部分标签，以让训练过程变得简单
# 仅保留了 “toxic”
train_df.drop(
    labels=["severe_toxic", "obscene", "threat", "insult", "identity_hate"],
    axis=1, inplace=True
)
print(train_df)

# 只保留1.2W条
# loc函数包含最后一条，而iloc则不包含最后一条
train_df = train_df.loc[:12000, :]

print(train_df.describe())
print(train_df.shape)

# 查看这部分数据中，最长的句子有多长
maximum_text_len = \
    max([len(str(text).split()) for text in train_df["comment_text"]])

print(maximum_text_len)


# 准备数据
x_train, x_valid, y_train, y_valid = train_test_split(
    # 为了与 Mr_Knownothing 大神的结果一致，random_state选42
    train_df["comment_text"].values, train_df["toxic"].values,
    test_size=0.2, shuffle=True, random_state=42,
    # 输入类数组对象, 默认值为None。如果不是None，则以分层方式拆分数据，并将其用作类标签。
    stratify=train_df.toxic.values
)
print(x_train.shape)
print(x_valid.shape)
print(y_train.shape)
print(y_valid.shape)


token = keras.preprocessing.text.Tokenizer(num_words=None)
max_len = 1500

# 根据训练样本习得词汇表
token.fit_on_texts(np.append(x_train, x_valid, axis=0))
x_train_seq = token.texts_to_sequences(x_train)
x_valid_seq = token.texts_to_sequences(x_valid)

# print(len(x_train_seq))
# print(x_train[0])
# print(x_train_seq[0])
# print(len(x_train_seq[0]))
# print(len(x_valid_seq))
# print(len(x_valid_seq[0]))

# 零填充
x_train_pad = keras.preprocessing.sequence.pad_sequences(
    x_train_seq, maxlen=max_len
)
x_valid_pad = keras.preprocessing.sequence.pad_sequences(
    x_valid_seq, maxlen=max_len
)

print(len(x_train_pad[0]))
print(len(x_valid_pad[0]))

# 获取 dict {词：下标} 字典
word_index = token.word_index
# 使用保存好的词汇表，方便快捷。太NICE
embedding_matrix = np.load("datasets/my_embedding_matrix.npy")

# 构建一个最简单的RNN来学习toxic
with strategy.scope():
    # 词汇表有很多单词，每个单词有一个下标值，
    # 但我们无法从中学习到词与词之间的联系。故使用词嵌入学习词与词之间的联系
    model = keras.Sequential([
        Embedding(input_dim=len(word_index) + 1, output_dim=300,
                  weights=[embedding_matrix], trainable=False,
                  input_length=max_len),
        # 该模型属于RNN中的多对一，整个序列输入后输出一个值
        SimpleRNN(100),
        Dense(1, activation="sigmoid")
    ], name="my_easy_rnn")
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    print(model.summary())

model.fit(x_train_pad, y_train, batch_size=256, epochs=3)


def roc_auc(predictions, target):
    '''
    This methods returns the AUC Score when given the Predictions
    and Labels
    '''

    fpr, tpr, thresholds = metrics.roc_curve(target, predictions)
    roc_auc = metrics.auc(fpr, tpr)
    return roc_auc


scores = model.predict(x_valid_pad)
print(f"Auc:{roc_auc(scores,y_valid)}")
