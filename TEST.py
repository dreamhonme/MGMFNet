# import os
#
# # 指定路径
# path = 'G:\code\DconnNet-mainT\CHASEDB1\img\Image_01L.jpg'
#
# # 检查路径是否存在
# if os.path.exists(path):
#     print(f"The path '{path}' exists.")
# else:
#     print(f"The path '{path}' does not exist.")

#为什么输出的是一样的呢？ 不应该一个水平一个垂直吗？

#import torch
#
# # 初始化平移矩阵
# W = 4
# H = 4
# NumClass = 1
# hori_translation = torch.zeros([1, NumClass, W, W])
# for i in range(W - 1):
#    hori_translation[:, :, i, i + 1] = 1.0  # 直接赋值1.0而不是创建新的tensor
# verti_translation = torch.zeros([1, NumClass, H, H])
# for j in range(H - 1):
#    verti_translation[:, :, j, j+1] = 1.0  # 直接赋值1.0而不是创建新的tensor
#
# # 打印平移矩阵
# print(hori_translation)
# print()
# print(verti_translation)
#
# import torch
# conn = torch.zeros([2, 4,4, 4]).cuda()
# print(conn)
# print(conn.shape)  # 输出: torch.Size([1, 8, 8, 8])
# print(conn.dtype)  # 输出: torch.float32 (默认的数据类型)


# import torch
#
# # 定义原始掩码
# mask = torch.tensor([[1, 2, 3],
#                      [4, 5, 6],
#                      [7, 8, 9]])
#
# # 将掩码扩展到一个批次中，模拟batch维度
# mask = mask.unsqueeze(0)  # 现在mask的形状是[1, 3, 3]
#
# # 创建一个全零的张量，形状为[1, 3, 2]，用于存储上移后的结果
# up = torch.zeros([1, 3, 3])
#
# # 将原始掩码的每一行向上移动一个位置
# # 这里选择了mask的第二行到最后一行，以及所有列
# # 并将结果赋值给up的第一行，以及所有行
# up[:,:3-1, :] = mask[:,1:3,:]
#
# print("Original mask:\n", mask)
# print("Mask after shifting up:\n", up)


# import torch

# # 假设 mask 和 up 是形状为 [3, 3] 的二维张量
# mask = torch.tensor([[1, 2, 3],
#                      [4, 5, 6],
#                      [7, 8, 9]], dtype=torch.float32)
# up = torch.tensor([[9, 8, 7],
#                    [6, 5, 4],
#                    [3, 2, 1]], dtype=torch.float32)

# # 计算 mask * up
# result = mask * up

# # 假设 conn 是形状为 [2, 8, 3, 3] 的四维张量
# conn = torch.zeros(1, 8, 3, 3, dtype=torch.float32)
# print(conn)

# # 假设 i = 0，将结果存储在 conn 的第 6 列
# i = 0
# conn[:, (i*8)+6, :, :] = result

# print("conn:\n", conn)


import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)

# import six
# print(six.__version__)

import apex
print(apex.__path__)

from apex import amp
print("Apex AMP is available.")