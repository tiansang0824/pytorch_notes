"""
练习线性层linear，顺便做了一个VGG16模型
"""
import torch
import torchvision.datasets
import torchvision.transforms
import torch.utils.data.dataloader

dataset = torchvision.datasets.CIFAR10(
    root='../dataset/',
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True
)

loader = torch.utils.data.DataLoader(
    dataset,
    64,
    True
)


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_features=196608, out_features=10)

    def forward(self, x):
        y = self.linear1(x)
        return y


n = Net()

for data in loader:
    imgs, labels = data
    print(f'>> imgs.shape: {imgs.shape}')
    # 变形
    # 这里采用了torch的flatten函数，该函数可以达到摊平数据的功能，但是不行像reshape一样指定维度。
    # x = torch.reshape(imgs, (1, 1, 1, -1))
    x = torch.flatten(imgs)  # 变形成一维数据
    print(f'>> x.shape: {x.shape}')
    # 模型参与
    y = n(x)
    print(f'>> y.shape: {y.shape}')
