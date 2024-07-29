import torch
import torchvision.datasets
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import DataLoader
from torch.nn import Module

cifar = torchvision.datasets.CIFAR10(
    root='../dataset',
    train=False,  # 选择测试集做训练
    transform=torchvision.transforms.ToTensor(),
    download=False
)

loader = DataLoader(
    dataset=cifar,
    batch_size=64,
    shuffle=True,
    drop_last=False
)


class Net(Module):
    def __init__(self):
        super().__init__()
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, x):
        y = self.maxpool1(x)
        return y


writer = SummaryWriter(log_dir='../logs/maxpooling_cifar')

n = Net()
step = 0
for data in loader:
    imgs, labels = data
    y = n(imgs)
    print(f'>> y.shape: {y.shape}')
    writer.add_images(tag=f'maxpooling', img_tensor=imgs, global_step=step)
    step += 1
