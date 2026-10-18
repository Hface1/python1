import torch
import numpy as np

#    tensor

#tensor by list
torch.tensor([1,2]), torch.tensor([[3,4],[5,6]])
#tensor by numpy
torch.from_numpy(np.array([[1,2],[3,4],[5,6]]))

print(torch.zeros(2,3))

print(torch.ones(2,3))

#填充一个两行三列的矩阵，元素全为7,becare[]
print(torch.full([2,3],7))

#对角矩阵
print(torch.eye(3,3))

#创建但不初始化
print(torch.empty(2,3))

#标准的正态分布，均值为0，方差是1
print(torch.randn(2,3))

#0-1均匀分布
print(torch.rand(2,3))

#1-9的整数均匀分布
print(torch.randint(1,10,[2,3]))

#自增函数
print(torch.arange(0,10,1))

#等差数列
print(torch.linspace(1,10,4))

#指定类型的创建    float32           int64
torch.FloatTensor([1,2,3,4,5,6]), torch.LongTensor([1,2,3,4,5,6])

#获得数据类型
torch.randn(2,3).dtype
torch.randn(2,3).shape

#转换数据类型
torch.randn(2,3).to(torch.float64).dtype



