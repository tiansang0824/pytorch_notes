"""
最大池化的目的一般是为了提取图片中的特征，同时减小图片数据量。

"""
import torch

X = torch.tensor(  # 创建输入数据
    data=[
        [1, 2, 0, 3, 1],
        [0, 1, 2, 3, 1],
        [1, 2, 1, 0, 0],
        [5, 2, 3, 1, 1],
        [2, 1, 1, 0, 1]
    ],
    dtype=torch.float32
)

x = torch.reshape(X, shape=(-1, 1, 5, 5))  # 变形
print(f'x.shape = {x.shape}')
print(f'>> x = \n{x}')


class Net(torch.nn.Module):
    """
    定义神经网络
    """

    def __init__(self):
        super().__init__()
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3, padding=0, ceil_mode=True)

    def forward(self, x):
        y = self.maxpool1(x)
        return y


net = Net()
y = net(x)
print(f'>> type(y): {type(y)}')
print(f'>> y: \n {y}')
