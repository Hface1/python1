import torch

a = torch.FloatTensor([[1,2,3],[4,5,6],[7,8,9]])

print(a.min())
print(a.max())
print(a.mean())#求平均
print(a.sum())#求和
print(a.prod())#求积
print(a.argmax())#求最大值的下标
print(a.argmin())#求最小值的下标



print(a.min(dim=0))
print(a.max(dim=0))
print(a.mean(dim=0))
print(a.sum(dim=0))
print(a.prod(dim=0))

#求1的范数
print(a.norm(1))
print(a.norm(1,dim=1))
#求2的范数
print(a.norm(2))
print(a.norm(2, dim=1))
#求前2个最小值
a.topk(3,dim=1,largest=True)
#求第2个小的值
a.kthvalue(2,dim=1)


