import torch

#生成数据函数
def get_rectanngle():
    import random

    #随机获得矩形的宽和高,值域0-1
    width = random.random()
    height = random.random()

    #判断
    fat = int(width >= height)

    return width, height,fat


get_rectanngle()

#定义数据集
class Dataset(torch.utils.data.Dataset):
    #对数据集的初始化，数据的加载处理等工作
    def __init__(self):
        pass


    #定义数据的条数
    def __len__(self):
        return 500


    #根据序号索引，来获得一条数据
    def __getitem__(self, index):
        width,height,fat = get_rectanngle()

        x = torch.FloatTensor([width,height])
        y = fat

        return x,y

dataset = Dataset()

#数据加载器，每八条一个批次，打乱数据顺序，不足八时丢弃尾数
loader = torch.utils.data.DataLoader(dataset,batch_size=8,shuffle=True,drop_last=True)

#看数据批次与看数据样例
len(loader),next(iter(loader))

#定义神经网络模型
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()      #sequential这个类意思是把多层的神经网络组合在一起，前后串联
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=2,out_features=32),     #输入2维度的数据，输出32维度的向量
            torch.nn.ReLU(),            #激活函数，让32维向量所有小于0的（负数）变为0，无参数
            torch.nn.Linear(in_features=32,out_features=32),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=32,out_features=2),    #2分类任务输出2维的向量作为结果
            torch.nn.Softmax(dim=1),  #让2个神经元的输出等于1，得到概率
        )

    #定义神经网络计算
    def forward(self,x):
        x = self.fc(x)

model = Model()




#训练
def train():
    #优化器，根据梯度来调整模型参数
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

    #计算loss函数
    loss_fn = torch.nn.CrossEntropyLoss()

    #让model进入train模式，开启dropout等功能，可不写
    model.train()

    #全量数据便历100轮
    for epoch in range(100):
        #按批次遍历loader中的数据
        for index,(x,y) in enumerate(loader):
            #模型计算
            output = model(x)

            #计算结果与y的差
            loss = loss_fn(output,y)

            #根据loss算模型的梯度
            loss.backward()

            #根据梯度调整模型的参数
            optimizer.step()

            #梯度归零，准备下一轮
            optimizer.zero_grad()
        if epoch % 20 == 0:
            #计算正确率
            acc = (output.argmax(dim=1) == y).sum().item() / len(y)
            print(epoch,loss.item(),acc)
    #训练结果存入磁盘
    torch.save(model,'model/3.model')
train()

#测试
#不计算模型梯度，节约计算资源
@torch.no_grad()
def test():
    #从磁盘里加载模型
    model = torch.load('model/3.model')
    #模型进入测试，关闭dropout等功能
    model.eval()
    #获得一批数据
    x,y = next(iter(loader))
    output = model(x)
    print(output,y)
    print(output == y)

test()











