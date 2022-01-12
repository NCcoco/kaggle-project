# 图像分类鸟类分类
325个鸟类的数据集。47332个训练图像，1625个测试图像（每个物种5个图像）和1625个验证图像（每个物种5个图像。  
所有图像都是jpg格式的224 X 224 X 3彩色图像。  
数据集包括一个训练集, 测试集和验证集. 每集包含 325 个子目录, 每个鸟种一个.   
如果使用 Keras ImageDataGenerator.flow from目录创建训练, 测试和有效数据生成器,数据结构很方便.  
数据集也包含一个文件 Bird Species.csv。这个 cvs 文件包含三列。filepaths 列包含图像文件的文件路径。labels 列包含与图像文件关联的类名。Bird Species.csv 文件如果使用读取df=pandas.birdscsv(Bird Species.csv) 将创建一个 Pandas 数据帧，然后可以将其拆分为训练df、测试df 和有效df 数据帧，以创建您自己的数据划分为训练、测试和有效数据集。  
注意：数据集中的测试和验证图像是手工选择的“最佳”图像，因此与创建自己的测试和验证集相比，使用这些数据集的模型可能会获得最高的准确度分数。然而，后一种情况在看不见的图像上的模型性能方面更准确。  
图像是从互联网搜索中按物种名称收集的。一旦下载了一个物种的图像文件，他们就会使用我开发的 python 重复图像检测器程序检查重复图像。删除所有检测到的重复项，以防止它们在训练、测试和验证集之间成为通用图像。  
之后，图像被裁剪，使鸟占据图像中至少 50% 的像素。然后将图像以 jpg 格式调整为 224 X 224 X3。裁剪确保当由 CNN 处理时，图像中有足够的信息来创建高度准确的分类器。即使是中等稳健的模型也应该达到 90% 的高范围内的训练、验证和测试准确度。对于每个物种，所有文件也从一个开始按顺序编号。所以测试图像被命名为 1.jpg 到 5.jpg。验证图像也是如此。训练图像也按“零”填充顺序编号。例如 001.jpg, 002.jpg ....010.jpg, 011.jpg .....099.jpg, 100jpg, 102.jpg 等。当与python文件函数和Keras flow from directory一起使用时，零的填充保留了文件顺序.  
训练集不平衡，每个物种有不同数量的文件。然而，每个物种至少有 120 个训练图像文件。这种不平衡并没有影响我的内核分类器，因为它在测试集上达到了 98% 以上的准确率。  
数据集中的一个显着不平衡是雄性物种图像与雌性物种图像的比率。大约 85% 的图像是男性，15% 是女性。典型的雄性颜色更加多样化，而一个物种的雌性通常是平淡的。因此，雄性和雌性图像可能看起来完全不同。几乎所有的测试和验证图像都取自该物种的雄性。因此，分类器在雌性物种图像上可能表现不佳。  

## 1. 加载图像数据
1. 下载数据集  
```
import tensorflow as tf
import pathlib  
# 以数据根目录为根目录，获取所有整个目录树
data_root = pathlib.Path('数据根目录')  
# 获取所有的目录，并以目录名称排序得到一个list （用目录作为标签）  
label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())  
# 为目录（标签）分配索引,得到一个字典对象，该对象以name为key，index为值
label_to_index = dict((name, index) for (index, name in enumerate(label_names))
# 创建一个列表，该列表包含所有文件所属标签的索引
# pathlib.Path是一个路径对象 [.parent.name] 则获取到上级目录的名称
all_image_labels = [label_to_index[pathlib.Path(path).parent.name] \
        for path in all_image_paths]]
# 加载及格式化图片  (自己的版本)
def load_image(img_path):  
    img_raw = tf.io.read_file(img_path)  
    img_tensor = tf.image.decode_image(img_raw)  
    # 这一步是将原图片格式化为符合model的图片
    # img_final = tf.image.resize(img_tensor, [224,224])
    img_final = tf.cast(img_final, dtype='float32')
    img_final = img_final/255.0
    return img_final
    
def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, [192, 192])
  image = tf.cast(image, dtype='float32')
  image /= 255.0  # normalize to [0,1] range
  return image
# 加载及格式化图片(官方版本)
def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)

# 构建数据集
def get_ds(all_image_paths):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
    # 由于这些数据集顺序相同，你可以将他们打包在一起得到一个(图片, 标签)对数据集：
    image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
    return image_label_ds

# 打乱数据集
image_label_ds.shuffle(buffer_size=image_count)

```


















