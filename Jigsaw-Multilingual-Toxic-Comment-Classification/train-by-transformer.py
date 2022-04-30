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
from tokenizers import BertWordPieceTokenizer


# 使用transformer
train_df = pd.read_csv("datasets/jigsaw-toxic-comment-train.csv")
valid_df = pd.read_csv("datasets/validation.csv")
test_df = pd.read_csv("datasets/test.csv")
# TODO: 这一步是干什么的？
sub = pd.read_csv("datasets/sample_submission.csv")


def fast_encode(texts, tokenizer, chunk_size=256, max_len=512):
    # 用于将文本编码为 BERT 输入的整数序列的编码器
    tokenizer.enable_truncation(max_length=max_len)
    tokenizer.enable_padding(max_length=max_len)
    all_ids = []

    for i in tqdm(range(0, len(texts), chunk_size)):
        text_chunk = texts[i:chunk_size + 1].tolist()
        encs = tokenizer.encode_batch(text_chunk)
        all_ids.extend([enc.ids for enc in encs])

    return np.array(all_ids)


# 超参数设置
AUTO = tf.data.experimental.AUTOTUNE
epochs = 3
batch_size = 16
max_len = 192

# Tokenization 分词器
tokenizer = transformers.DistilBertTokenizer.from_pretrained(
    'distilbert-base-multilingual-cased'
)
# 保存这个预训练的分词器
tokenizer.save_pretrained('.')
exit()
fast_tokenizer = BertWordPieceTokenizer('tmp/vocab.txt', lowercase=False)
print(fast_tokenizer)
