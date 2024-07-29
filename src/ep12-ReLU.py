import torch
from torch.nn import ReLU

x = torch.tensor(
    data=[
        [1, -0.5],
        [-1, 3]
    ],
    dtype=torch.float32
)

# 数据变形
x = torch.reshape(x, (-1, 1, 2, 2))


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu1 = ReLU()

    def forward(self, x):
        y = self.relu1(x)
        return y


n = Net()
y = n(x)
print(f'>> y.shape: {y.shape}')
print(f'>> y.value: \n{y}')
