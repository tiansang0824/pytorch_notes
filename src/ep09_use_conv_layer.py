"""
以conv2d_layer为例，说明卷积层的使用方法。

"""
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.nn import Conv2d
from torch.utils.tensorboard import SummaryWriter

# 导入数据集
cifar_dataset = torchvision.datasets.CIFAR10(
    root='../dataset',
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor()
)

# 创建DataLoader
cifar_loader = DataLoader(cifar_dataset, batch_size=64, shuffle=True, drop_last=False)


class Net(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, padding=0, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        return x


net = Net()
print(f'network frame: \n{net}')

writer = SummaryWriter(log_dir='../logs/nn')

i = 0
for data in cifar_loader:
    imgs, labels = data
    print(f'>> imgs.shape: {imgs.shape}')
    output = net(imgs)
    print(f'>> output.shape: {output.shape}')
    # output的尺寸是6个channel，但是writer.add_images要求的是3个channel，所以不能直接用于显示图片。
    # 所以先进行一次变换
    output = torch.reshape(output, (-1, 3, 30,30))
    print(f'>> output.shape: {output.shape}\n')

    writer.add_images(tag='nn_images', img_tensor=output, global_step=i)  # 使用Tensorboard显示图片
    i += 1
