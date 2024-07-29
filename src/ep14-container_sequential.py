"""
写一个用于分类CIFAR数据集的简单神经网络，且中途利用sequential实现神经网络。
"""
import torch
from torch.nn import Conv2d, MaxPool2d, Linear, Flatten


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 输入数据格式为(3,32,32) 第一个值为通道数目
        # 已知输入格式和输出格式，通过卷积公式计算padding和stride
        self.conv1 = torch.nn.Conv2d(3, 32, 5, padding=2)
        # 第一次卷积后的格式为(32,32,32)
        self.maxpooling1 = torch.nn.MaxPool2d(kernel_size=2)
        # 池化后的格式为(32,16,16)
        self.conv2 = torch.nn.Conv2d(32, 32, 5, padding=2)
        # 卷积后的尺寸是(32,16,16)
        self.maxpooling2 = torch.nn.MaxPool2d(kernel_size=2)
        # 池化后的尺寸是(32,8,8)
        self.conv3 = torch.nn.Conv2d(32, 64, 5, padding=2)
        # 卷积后的尺寸是(64,8,8)
        self.maxpooling3 = torch.nn.MaxPool2d(kernel_size=2)
        # 池化后是(64,4,4)
        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(in_features=1024, out_features=64)
        self.linear2 = torch.nn.Linear(64, 10)

        # 如果写一个类似下面的Sequential的话，就可以把前面的定义删掉了。
        self.model1 = torch.nn.Sequential(
            Conv2d(3,32,5,padding=2),
            MaxPool2d(2),
            Conv2d(32,32,5,padding=2),
            MaxPool2d(2),
            Conv2d(32,64,5,padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024,64),
            Linear(64,10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpooling1(x)
        x = self.conv2(x)
        x = self.maxpooling2(x)
        x = self.conv3(x)
        x = self.maxpooling3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        y = self.linear2(x)
        return y


n = Net()
print(f'>> print Net obj: n = {n}')


# 检验网络（尺寸）数据是否合理
input = torch.ones((64, 3, 32, 32))
output = n(input)
print(f'>> output.shape: {output.shape}')
print(f'>> output: {output}')
