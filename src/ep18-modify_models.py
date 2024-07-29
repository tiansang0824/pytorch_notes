# @File：    ep18-modify_models.py
# @Author:  tiansang0824
# @Time:    2024/7/20 10:12
# @Description: 
# 用于练习对模型的集成和修改。
import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn

# 下载数据集
# 【注意】该数据集总大小约有147+GB，且已经不在提供公开的下载，
# 因此本文件中不再下载该内容，后续仅通过修改`weights`参数，观察适应不同数据集后，模型的参数区别。
# train_data = torchvision.datasets.ImageNet(
#     '../dataset/',
#     'train',
#     download=True,
#     transform=torchvision.transforms.ToTensor()
# )


# 创建一个默认的vgg16模型
vgg16_model = torchvision.models.vgg16()
# 通过配置预设定的权重参数weights，使得创建的模型可以直接应用于ImageNet数据集
vgg16_imageNet = models.vgg16(weights='DEFAULT')

print(f'okk')

# 观察最后的输出层，该层的输出特征数为1000，即最后会对1000个不同的类别计算概率。
print(f'>> vgg16_imageNet_structure: \n{vgg16_imageNet}')
print(f'>> vgg16_model_structure: \n{vgg16_model}')
print(f'\n\n>> divide line << \n\n')
# 下面尝试对模型进行修改，使其能够适应CIFAR训练集数据
cifar_dataset = torchvision.datasets.CIFAR10(
    root='../dataset',
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

# 然后修改模型
# vgg16最后的输出有1000个特征，但是CIFAR只有10个类别，
# 因此需要修改vgg6的输出层输出特征数量，
# 或者在原有的输出层后面再加一个输入1000、输出10的层。

# 第一种方法，先加一个新的层。
# 在最后添加了一个新的linear层，输入特征为1000，输出特征为10
# vgg16_imageNet.add_module('add_linear', nn.Linear(1000, 10))  # 这样可以在模型最后添加新的模型
# 下面的方法可以将新的linear层添加到模型的classifier的最后一层中。
vgg16_imageNet.classifier.add_module('new_linear', nn.Linear(1000, 10, bias=True))
print(f'>> vgg after added linear layer: \n{vgg16_imageNet}')
print(f'\n\n>> ========== divide line ========== << \n\n')

# 第二种方法，修改原有的神经层
# 通过索引的方法修改神经层
vgg16_model.classifier[6] = nn.Linear(4096, 10, bias=True)
print(f'>> vgg_model after modified linear layer: \n{vgg16_model}')
