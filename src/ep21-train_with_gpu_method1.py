# @File：    ep21-train_with_gpu_method1.py
# @Author:  tiansang0824
# @Time:    2024/7/20 17:42
# @Description: 
# 使用GPU训练模型的方法一：补充cuda()函数。
# 本文件中采用的模型同ep20。

import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data.dataloader as data_loader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.nn as nn

"""
可以调用CPU参与计算的内容包括：
- 模型实例
- 损失函数
- 数据（训练中通过DataLoader获取的数据内容）

设置方法：
在实例后调用cuda()函数
"""

# 训练数据集
train_data = torchvision.datasets.CIFAR10(
    root='../dataset',
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

# 测试数据集
test_data = torchvision.datasets.CIFAR10(
    root='../dataset',
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

# 信息展示
train_data_size = len(train_data)
test_data_size = len(test_data)
print(f'>> len of train data: {train_data_size}; len of test data: {test_data_size}')

# 利用DataLoader加载数据集
train_loader = data_loader.DataLoader(
    train_data,
    64,
    True,
    drop_last=False
)

# 加载测试数据集
test_loader = data_loader.DataLoader(
    test_data,
    64,
    True,
    drop_last=False
)

"""
类定义内容建议新建一个文件单独保存
"""


class TianNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


# 创建神经网络
tiannet = TianNet()
# 调用GPU参与训练。
if torch.cuda.is_available():
    tiannet = tiannet.cuda()

# 定义损失函数
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()  # 使用GPU

# 定义优化器
learning_rate = 1e-2
optimizer = optim.SGD(tiannet.parameters(), lr=learning_rate)

# 设置训练网络的参数
total_train_step = 0  # 记录训练次数
total_test_step = 0  # 记录测试次数
epochs = 10  # 训练轮数

# 添加TensorBoard
writer = SummaryWriter('../logs/cifiar_classification')

# 开始训练
for i in range(epochs):
    print(f'>> epoch {i} starts...')

    # 训练步骤开始
    tiannet.train()  # 【可选】设置模型到训练状态
    for data in train_loader:
        # 获取数据
        imgs, labels = data
        # 采用GPU参与训练
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            labels = labels.cuda()

        # 放入模型
        y_hat = tiannet(imgs)  # 计算y_hat

        # 训练参数
        loss = loss_fn(y_hat, labels)  # 计算损失
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播
        optimizer.step()  # 优化参数

        # 记录新的训练次数
        total_train_step += 1

        # 保存数据
        if total_train_step % 100 == 0:
            print(f'\ttrain step {total_train_step} times, loss: {loss.item()}')
            writer.add_scalar('train loss', loss.item(), global_step=total_train_step)

    # 测试步骤开始
    tiannet.eval()  # 【可选】设置模型到测试状态
    # 准确率记录
    total_accuracy = 0  # 总正确率
    total_test_loss = 0  # 总损失值
    with torch.no_grad():  # 这个函数关闭了梯度调整的设置
        for data in test_loader:
            imgs, targets = data  # 获取数据
            # 采用GPU参与训练
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()

            outputs = tiannet(imgs)  # 使用模型预测输出

            loss = loss_fn(outputs, targets)  # y_hat与y进行对比，计算损失函数
            total_test_loss = total_test_loss + loss.item()  # 损失函数加和，用于比较每一轮训练中的损失函数变化

            # 计算正确率
            accuracy = (outputs.argmax(1) == targets).sum()  # 一次测试中正确的数量
            total_accuracy = total_accuracy + accuracy  # 总体正确数量加和

    # 输出结果
    print(f'>> total loss on test dataset: {total_test_loss}')  # 打印损失
    print(f'>> total accuracy on test dataset: {total_accuracy / test_data_size}')  # 正确数量减去总体数量

    # 保存结果
    writer.add_scalar('test loss', total_test_loss, global_step=total_test_step)  # 添加记录
    writer.add_scalar('test accuracy', total_accuracy / test_data_size, global_step=total_test_step)  # 添加正确率记录

    # 更新数据
    total_test_step += 1  # 更新步骤值

    # 保存每一轮训练后的模型
    torch.save(tiannet.state_dict(), f'../saved_models/cifar_model/tiannet_epoch_{i}.pth')
    print('>> model saved.')

writer.close()
