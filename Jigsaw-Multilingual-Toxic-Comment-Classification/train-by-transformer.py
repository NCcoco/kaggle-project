import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.api._v2 import keras
from keras.api._v2.keras.layers import Dense, Input
from keras.api._v2.keras.optimizers import Adam
from keras.api._v2.keras.models import Model
from keras.api._v2.keras.callbacks import ModelCheckpoint
import transformers

# WordPiece 可以缩短词汇表，将词的前缀与后缀分割开，以保证不同时态的同一个词被记录为一个词
from tokenizers.implementations.bert_wordpiece import BertWordPieceTokenizer


# 使用GPU
strategy = tf.distribute.get_strategy()

# 使用transformer
train_df = pd.read_csv("datasets/jigsaw-toxic-comment-train.csv")
valid_df = pd.read_csv("datasets/validation.csv")
test_df = pd.read_csv("datasets/test.csv")
# TODO: 这一步是干什么的？
sub = pd.read_csv("datasets/sample_submission.csv")


def fast_encode(texts, tokenizer, chunk_size=256, max_len=512):
    # 用于将文本编码为 BERT 输入的整数序列的编码器
    tokenizer.enable_truncation(max_length=max_len)
    tokenizer.enable_padding(length=max_len)
    all_ids = []

    for i in tqdm(range(0, len(texts), chunk_size)):
        text_chunk = texts[i:chunk_size + i].tolist()
        encs = tokenizer.encode_batch(text_chunk)
        all_ids.extend([enc.ids for enc in encs])

    return np.array(all_ids)


# 超参数设置
AUTO = tf.data.experimental.AUTOTUNE
epochs = 3
batch_size = 64
max_len = 192

# Tokenization 分词器
# tokenizer = transformers.DistilBertTokenizer.from_pretrained(
#     'distilbert-base-multilingual-cased'
# )
# 保存分词器
# tokenizer.save_pretrained('tmp/')
# exit()
fast_tokenizer = BertWordPieceTokenizer('tmp/vocab.txt', lowercase=False)

x_train = fast_encode(train_df.comment_text.astype(str),
                      fast_tokenizer, max_len=max_len)
print(x_train.shape)
x_valid = fast_encode(valid_df.comment_text.astype(str),
                      fast_tokenizer, max_len=max_len)
x_test = fast_encode(test_df.content.astype(str),
                     fast_tokenizer, max_len=max_len)

y_train = train_df.toxic.values
y_valid = valid_df.toxic.values


train_dataset = (tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(2048).batch(batch_size)
                 .prefetch(AUTO))

valid_dataset = (tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
                 .batch(batch_size).cache().prefetch(AUTO))

test_dataset = (tf.data.Dataset.from_tensor_slices(x_test).batch(batch_size))


def build_model(transformer, max_len=512):
    # 定义输入数据的格式
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32,
                           name="input_word_ids")
    sequence_output = transformer(input_word_ids)[0]
    # TODO: 这里没看懂
    cls_token = sequence_output[:, 0, :]

    out = Dense(1, activation='sigmoid')(cls_token)

    model = Model(inputs=input_word_ids, outputs=out)
    model.compile(optimizer=Adam(lr=0.001),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model


with strategy.scope():
    transformer_layer = (
        transformers.TFDistilBertModel
            .from_pretrained('distilbert-base-multilingual-cased',
                             trainable=False)
    )
    model = build_model(transformer_layer, max_len=max_len)


print(model.summary())

n_steps = x_train.shape[0]
train_history = model.fit(
    train_dataset,
    batch_size=batch_size,
    validation_data=valid_dataset, epochs=epochs
)

sub['toxic'] = model.predict(test_dataset, verbose=1)
sub.to_csv('submission.csv', index=False)





