# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as keras

print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist
print(fashion_mnist)
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
train_images = np.expand_dims(train_images, axis=3)

train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.models.Sequential([
    keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu'),
    keras.layers.Flatten(input_shape=(28, 28, 32)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=10)
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)
predictions = model.predict(test_images)
print(np.argmax(predictions[0]))
print(test_labels[0])
