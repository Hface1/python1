import torch


#       索引tensor

#创建测试数据
#4张图片 ，3通道 ，28*28的像素尺寸
#3通道一般指的是三元色，即彩色图片，1通道即黑白图片

#               图片数，通道数，尺寸
a = torch.randn(4,3,28,28)

print(a.shape)

#查看第0张图片
print(a[0].shape)

#查看第0张图片的的0个通道
print(a[0,0].shape)

#查看第0张图片的的0个通道的第2行
print(a[0,0,2].shape)

#查看第0张图片的的0个通道的第2行第4列
print(a[0,0,2,4].shape)

#       切片

#查看0-1张图片       左闭右开
print(a[0:2].shape)
print(a[:2, :2].shape)
print(a[:2,:2,-5:25].shape)
print(a[:,:,:,::2].shape)

#用...表示多个被省略的
#取所有图片的第0-2列    和取行    和取第二张图
print(a[...,:2].shape)
print(a[...,:2,:].shape)
print(a[2,...].shape)







