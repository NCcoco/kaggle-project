import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa


#分类数，我们要使用cifar100这个数据集，这里有100种分类，所以这里我们设置100
num_classes = 100
input_shape = (32, 32, 3) #输入图片的大小（h, w, c)

(x_train, y_train),(x_test,y_test) = keras.datasets.cifar100.load_data()

print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")

learning_rate = 0.001 #学习率
weight_decay = 0.001 #权重衰变率
batch_size = 64 #每次读取的图片数
num_epochs = 100 #训练的总循环次数
image_size = 64 #因为要对图片数据进行增强，把原先32的图片变成64的
patch_size = 6 #我们要对图片进行分割 分割后的图片大小是6
num_patches = (image_size // patch_size) ** 2 #大小是144 就是图片被分割后的小块数量
projection_dim = 64 #这个也就是我们每个分割后小图片的embedding_dim.
num_heads = 8 #多头注意力机制里面的多头数
# transformer_units 这个是在mlp增加模型非线性时增加的神经元个数，这个增加非线性主要用在transformer模块里
transformer_units = [
    projection_dim * 2,
    projection_dim,
]
transformer_layers = 8 #transformer模块的层数，
#这个跟transformer_units一样也是增加非线性，这个是用在transformer之后，进行全连接时增加非线性
mlp_head_units = [2048, 1024]

data_augmentation = keras.Sequential(
    [
        layers.Normalization(), #归一化
        layers.Resizing(image_size, image_size), #调整图片大小
        layers.RandomFlip("horizontal"), #随机翻转图像, "horizontal"是左右翻转
        layers.RandomRotation(factor=0.02), #随机旋转图像 表示顺时针和逆时针旋转的下限和上限。正值表示逆时针旋转，负值表示顺时针旋转。当表示为单个浮点数时，此值用于上限和下限。
        layers.RandomZoom(
            height_factor=0.2, width_factor=0.2 #随机缩放图像
        ),
    ],
    name="data_augmentation",
)

def mlp(x, hidden_units, dropout_rate): #主要作用就是增加非线性
    for units in hidden_units: #因为hidden_units是一个列表，我们要进行for循环一下。
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size  # 这里patch_size我们设置为6

    def call(self, images):  # 这里的images是做了数据增强的image （batch_size, 72,72,3)
        batch_size = tf.shape(images)[0]  # 得到batch_size大小
        # tf.image.extract_patches 就是把72*72的图片分割成横竖都是12个大小6*6*3 108个特征的图片
        # (batch_size, 12, 12, 108)
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]  # 取出被分割后小图片的所有特征
        # 改变数据形态，就是12*12
        patches = tf.reshape(patches, [batch_size, -1,
                                       patch_dims])  # （batch_size, 144, 108)
        return patches

import matplotlib.pyplot as plt

plt.figure(figsize=(4, 4))
image = x_train[np.random.choice(range(x_train.shape[0]))] #随机选取一个图片数据
plt.imshow(image.astype("uint8")) #显示图片
plt.axis("off")

#给图片做数据增强，
resized_image = tf.image.resize(
    tf.convert_to_tensor([image]), size=(image_size, image_size)
)
patches = Patches(patch_size)(resized_image) #对图片进行分割
print(f"Image size: {image_size} X {image_size}")
print(f"Patch size: {patch_size} X {patch_size}")
print(f"Patches per image: {patches.shape[1]}")
print(f"Elements per patch: {patches.shape[-1]}")

n = int(np.sqrt(patches.shape[1])) # n=12
plt.figure(figsize=(4, 4))
for i, patch in enumerate(patches[0]): #patches[0] (144, 108)
    ax = plt.subplot(n, n, i + 1)
    patch_img = tf.reshape(patch, (patch_size, patch_size, 3)) #(6*6*3)=108
    plt.imshow(patch_img.numpy().astype("uint8"))
    plt.axis("off")


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        # num_patches = (image_size // patch_size) ** 2 #大小是144 就是图片被分割后的小块数量
        # projection_dim = 64 我们要进行线性变换后的神经元数量
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        # 下面是位置编码
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded  # (batch_size, 144, 64)


def create_vit_classifier():
    # 建立输入层，输入的shape（32，32，3）这时图片的原始大小
    inputs = layers.Input(shape=input_shape)
    # 对图片数据进行数据增强
    augmented = data_augmentation(inputs)
    # 对增强后的图片进行分割
    patches = Patches(patch_size)(augmented)
    # 对分割后的图片数据进行embedding
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # 创建多个Transformer结构的层。
    for _ in range(transformer_layers):
        # 先进行第一个归一化
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # 创建一个多头注意力层
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # 进行一次残差链接
        x2 = layers.Add()([attention_output, encoded_patches])
        # 进行第二个归一化
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # 进行mlp进行线性增强
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # 在进行一次残差链接
        encoded_patches = layers.Add()(
            [x3, x2])  # (batch_size, num_patches, projection_dim)

    # 对Transformer最后输出的结果进行归一化
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    # 把数据展平变成（batch_size, num_patches*projection_dim)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)

    # 在对数据进行一下非线性增强
    features = mlp(representation, hidden_units=mlp_head_units,
                   dropout_rate=0.5)

    # 生成分类 （batch_size, num_classes)
    logits = layers.Dense(num_classes)(features)

    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model

def run_experiment(model):
    #建立优化这里我们实现权重衰减的adamw的优化器
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )
    #进行编译
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )
    #建立检察点
    checkpoint_filepath = "tmp/checkpoint"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,

    )
    #进行训练模型拟合
    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.1,
        callbacks=[checkpoint_callback],
    )
    #加载训练后的模型
    model.load_weights(checkpoint_filepath)
    _, accuracy, top_5_accuracy = model.evaluate(x_test, y_test) #对测试数据进行预测
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

    return history


vit_classifier = create_vit_classifier()
history = run_experiment(vit_classifier)
