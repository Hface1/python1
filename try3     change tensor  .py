import torch

#view,reshape维度变换

a = torch.randn(4,1,28,28)
print(a.shape)

print(a.reshape(4,784).shape)#打平为一维度的向量

print((a.reshape(4,784)).reshape(4,28,28).shape)

#view与reshape用法一致，张量内存连续用view ，其他可reshape

# unsqueeze插入一个维度

#2*2的tensor
b = torch.randn(2,2)
print(b.shape)

#插入维度在第0维  与倒数第1维
print(b.unsqueeze(0).shape)
print(b.unsqueeze(1).shape)
print(b.unsqueeze(-1).shape)

#squeeze  删除维度，只能删除为1的维度
c = torch.randn(1,2,2,1)

print(c.shape)
#删除第1维度  倒数第2维度  不填自动删除所有为1的维度
print(c.squeeze(1).shape)
print(c.squeeze(-1).shape)
print(c.squeeze().shape)

#repeat 复制tensor的维度

#0维度上复制2次  1维度复制3次
print(torch.randn(2,2).repeat(2,3).shape)

#维度交换
#t转置  只操作2维度tensor
print(torch.randn(2,2).t().shape)
#transpose维度交换，指定维度，只可以两两交换
print(torch.randn(1,2,3,4).transpose(0,3).shape)

#permute维度交换，输入尉当户的顺序
print(torch.randn(2,3,4,5).permute(1,0,3,2).shape)

#手动broadcast

print(b.expand_as(a).shape)
print(c.expand_as(b).shape)











