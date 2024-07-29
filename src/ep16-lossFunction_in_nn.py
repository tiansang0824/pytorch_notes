import torch
from torch.nn import Module, MaxPool2d, Conv2d, Flatten, Linear
from torch.nn import CrossEntropyLoss
import torchvision
import torchvision.transforms as transforms
import torch.utils.data.dataloader as dataloader

cifar_dataset = torchvision.datasets.CIFAR10(
    root='../dataset',
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

loader = dataloader.DataLoader(
    dataset=cifar_dataset,
    batch_size=64,
    shuffle=True,
    drop_last=False
)


class Net(Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        y = self.model(x)
        return y


n = Net()
loss_fn = CrossEntropyLoss()
for data in loader:
    imgs, labels = data
    outputs = n(imgs)
    print(f'>> outputs.shape: {outputs.shape}')
    print(f'>> labels.shape: {labels.shape}')
    print(f'>> outputs[0]: {outputs[0]}; labels[0]: {labels[0]}')
    loss = loss_fn(outputs, labels)
    print(f'>> loss: {loss}')
    loss.backward()

    print('\n')
