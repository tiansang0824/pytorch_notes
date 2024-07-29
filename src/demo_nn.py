import torch.nn as nn
import torch


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output


my_model = MyModel()
x = torch.Tensor([1.0])
output = my_model(x)
print(output)
