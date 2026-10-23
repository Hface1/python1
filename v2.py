import torch
import torch.nn as nn
from torchvision import datasets, transforms

torch.manual_seed(0)

#数据（函数化）
def load_data():
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    """读数据，并把像素做标准化。

        Normalize(mean, std) 算的是 (x - mean) / std。
        0.1307 和 0.3081 是 MNIST 全部像素统计出来的均值/标准差（不是随便编的）。
        归一化后数据大致变成"均值 0、标准差 1"，各维度尺度一致，梯度下降更好走。
        """
    train_dataset = datasets.MNIST(
    root='./data', train=True, transform=tf, download=True
)
    test_dataset = datasets.MNIST(
    root='./data', train=False, transform=tf, download=True
)

    x = torch.stack([image for image,_ in train_dataset]).reshape(-1,784)
    y = torch.tensor([label for _,label in train_dataset])

    x_test = torch.stack([image for image,_ in test_dataset]).reshape(-1,784)
    y_test = torch.tensor([label for _,label in test_dataset])

    return x,y,x_test,y_test

def build_model():
    return nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )

#分批训练
"""分批训练。

    为什么一定要分批？
      全批量（V1）：每个 epoch 只用 60000 张图算一次梯度、更新 1 次。
                   100 个 epoch = 100 次更新。参数总共才被"调"了 100 下。
      分批（V2）  ：每 64 张算一次梯度、更新一次。
                   一个 epoch 有 60000/64 ≈ 938 次更新，5 个 epoch 就是 4690 次。
                   同样的时间，学习次数多了近 50 倍。

    第二个好处：内存。全批量要把 60000x784 个浮点数一起塞进计算图，
    分批只需要 64x784，内存占用小两个数量级，大数据集才装得下。
    """
def train(model,x,y,batch_size = 64,lr = 0.001,epochs = 10):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        prem = torch.randperm(len(x))#生成一个0到60000-1的随机排序
        x,y = x[prem],y[prem]#随索引来达到随机取值

        total_loss = 0
        for start in range(0,len(x),batch_size):
            xb = x[start:start+batch_size]
            yb = y[start:start+batch_size]

            optimizer.zero_grad()
            loss = loss_fn(model(xb),yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()*len(xb)#最后一批是32，比直接乘上64要准确
#注意loss得到的值是平均值，乘上样本数为了后面的加权平均
        print(f"第{epoch +1}轮  平均loss = {total_loss/len(x):.4f}")

#评估
def evaluate_model(model,x,y):
    with torch.no_grad():
        pred = model(x).argmax(dim=1)
        return (pred == y).float().mean().item()

#整合上述
if __name__ == "__main__":
    torch.manual_seed(0)
    x,y,x_test,y_test = load_data()
    model = build_model()
    train(model,x,y,batch_size = 64,lr = 0.001,epochs = 10)
    print(f"测试集的准确度；{evaluate_model(model,x_test,y_test)*100:.2f}%")

