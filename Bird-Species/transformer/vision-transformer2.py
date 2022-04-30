import torch
import torch.nn as nn
import torch.optim as optim
from vit_pytorch import ViT
from vit_jax import models


def load_checkpoint(checkpoint, model, optimizer):
    print("=> 加载checkpoint model")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

# 定义所需超参数
epoch = 1
batch_size = 256
image_size = 224
patch_size = 32
num_classes = 325
dim = 1024
depth = 6
learning_rate = 0.001


# 初始化模型
# 传入参数的意义：
# image_size：输入图片大小。
# patch_size：论文中 patch size： [公式] 的大小。
# num_classes：数据集类别数。
# dim：Transformer的隐变量的维度。
# depth：Transformer的Encoder，Decoder的Layer数。
# heads：Multi-head Attention layer的head数。
# mlp_dim：MLP层的hidden dim。
# dropout：Dropout rate。
# emb_dropout：Embedding dropout rate。

model = models.get_model("ViT-B_32", num_classes=num_classes)
optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)
load_checkpoint(torch.load("imagenet21k_ViT-B_32.npz"), model, optimizer)

print(model)


for param in model.parameters():
    param.requires_grad = False

exit()
img = torch.randn(1, 3, 256, 256)
# optional mask, designating which patch to attend to
mask = torch.ones(1, 8, 8).bool()

preds = model(img)  # (1, 1000)
print(preds.shape)
