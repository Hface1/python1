import torch

#tensor的拼接与拆分
d = torch.rand(4,32,8)
f = torch.rand(5,32,8)
#cat拼接，dim是指定要拼接的维度
g = torch.cat([d,f],dim=0).shape

print(g)

#stack   会创造一个新的维度
a = torch.rand(4,32,8)
b = torch.rand(4,32,8)

torch.stack([a,b],dim=0).shape

#拆分
#split 拆分

a = torch.rand(4,32,8)

_1,_2 = a.split(2,dim=0)
print(_1.shape)
print(_2.shape)
g = torch.rand(4,32,8)
_3,_4,_5 = g.split([1,2,1],dim=0)
print(_3.shape)
print(_4.shape)
print(_5.shape)

#chunk拆分，在0维度上拆分
_6,_7 = g.chunk(2,dim=0)
print(_6.shape)
print(_7.shape)

#四则运算
a = torch.rand(4,32,8)
b = torch.rand(4,32,8)

print(a+b)
print(a-b)
print(a*b)
print(a//b)
print(a%b)
print(a**b)

c = torch.FloatTensor([[1,2,3],[4,5,6],[7,8,9]])
d = torch.FloatTensor([[1,2,3],[4,5,6],[7,8,9]])
print(c@d)

print(c.exp())
print(c.log())
print(c.sin())
print(c.cos())
print(c.tan())
print(c.tanh())

print(a>b)
print(a<b)
print(a==b)
print(a!=b)


c = torch.FloatTensor([1.5])
#向下取整
print(c.floor())
#向上取整
print(c.ceil())
#四舍五入
print(c.round())

print(c.abs())



