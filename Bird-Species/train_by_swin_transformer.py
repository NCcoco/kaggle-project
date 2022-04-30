import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, losses
import tensorflow_hub as hub
from util.util import load_datasets


model = hub.load("https://tfhub.dev/rishit-dagli/swin-transformer/1")
model = tf.keras.Sequential([
  tf.keras.layers.Lambda(lambda data: tf.keras.applications.imagenet_utils.preprocess_input(tf.cast(data, tf.float32), mode="torch"), input_shape=[*IMAGE_SIZE, 3]),
  SwinTransformer('swin_tiny_224', include_top=False, pretrained=True),
  tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])
model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
              loss=losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

