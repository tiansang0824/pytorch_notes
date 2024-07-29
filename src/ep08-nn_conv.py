"""
本代码用于联系二维输入的卷积神经网络的简单操作
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

X = torch.tensor(  # 创建输入数据
    data=[
        [1, 2, 0, 3, 1],
        [0, 1, 2, 3, 1],
        [1, 2, 1, 0, 0],
        [5, 2, 3, 1, 1],
        [2, 1, 1, 0, 1]
    ]
)

conv_kernel = torch.tensor(
    data=[
        [1, 2, 1],
        [0, 1, 0],
        [2, 1, 0]
    ]
)

print(X.shape)
print(conv_kernel.shape)

# torch的尺寸变换
# 为什么变形？变成什么形？
# 可以参考：https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html#torch.nn.functional.conv2d
# 中对conv2d的参数要求
X = torch.reshape(X, (1, 1, 5, 5))  # 对输入X进行变形
print(X.shape)
conv_kernel = torch.reshape(conv_kernel, (1, 1, 3, 3))  # 对kernel进行变形
print(conv_kernel.shape)

output = F.conv2d(input=X, weight=conv_kernel, stride=1)  # 利用卷积核对输入X进行卷积操作
print(f'>> default conv output: \n{output}')  # 输出卷积结果

output2 = F.conv2d(input=X, weight=conv_kernel, stride=2)  # 设置stride可以修改卷积核每次移动的步长
print(f'>> output_stride=2: \n{output2}')

output_with_padding = F.conv2d(input=X, weight=conv_kernel, padding=1, stride=1)  # 设置padding=1可以在输入图像外围添加一层空值
print(f'>> output_with_padding: \n{output_with_padding}')

