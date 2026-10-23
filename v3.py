#防止数据泄漏，训练集周切出10%来作为验证集
#用测试集来防止过拟合导致的“虚高准确度”
#学习率从大到小，学习率退火

import copy

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms

torch.manual_seed(0)

def load_data():
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,)),
        #注意Normalize是逐通道来处理的，这里的通道数为1，但其依然应该是一组的数据，
        #所以要元组，有几个通道，元组就要有几个元数
    ])
    train_all = datasets.MNIST("./data", train=True,
                               download=True, transform=tf)
    test = datasets.MNIST("./data", train=False,
                          download=True, transform=tf)

    x = torch.stack([img for img,_ in train_all]).reshape(-1,784)
    y = torch.tensor([label for _,label in train_all])
    x_train, y_train = x[:54000], y[:54000]
    x_val, y_val = x[54000:], y[54000:]

    x_test = torch.stack([img for img,_ in test]).reshape(-1,784)
    y_test = torch.tensor([label for _,label in test])
    return (x_train,y_train), (x_val,y_val), (x_test,y_test)

def build_model():
    return nn.Sequential(
        nn.Linear(784,512),
        nn.ReLU(),
        nn.Linear(512,128),
        nn.ReLU(),
        nn.Linear(128,10),
    )

def evaluate(model,x,y):
    with torch.no_grad():
        pred = model(x).argmax(dim=1)
        return (pred == y).float().mean().item()

def train(model,x,y,x_val,y_val,batch_size = 64,lr = 0.1,epochs = 10):
    optimizer = torch.optim.SGD(model.parameters(),lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    scheduler = CosineAnnealingLR(optimizer,T_max=epochs)

    best_acc = 0.0
    best_state = None   #用来存最好的那一轮参数快照

    for epoch in range(epochs):
        perm = torch.randperm(len(x))
        x, y = x[perm], y[perm]

        for start in range(0,len(x),batch_size):
            xb = x[start:start+batch_size]
            yb = y[start:start+batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(xb),yb)
            loss.backward()
            optimizer.step()

        scheduler.step()#更新学习率（放在每个epoch之后）

        #监考，看验证集
        train_acc = evaluate(model,x,y)
        val_acc = evaluate(model,x_val,y_val)
        print(f"第{epoch + 1}轮  lr={scheduler.get_last_lr()[0]:.4f}"
              f"训练acc={train_acc*100:.2f}%  验证acc={val_acc*100:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc

            # 必须 deepcopy！state_dict 里存的是"张量的引用"，
            # 不深拷贝的话，后面参数继续变，这个"快照"会跟着一起变，等于白存。
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    print(f"\n最佳验证集准确率：{best_acc*100:.2f}%(参数已经滚回到这一轮)")

if __name__ == '__main__':
    torch.manual_seed(0)
    (x_train,y_train), (x_val,y_val), (x_test,y_test)= load_data()
    model = build_model()
    train(model,x_train,y_train,x_val,y_val,epochs = 10)

    print(f"测试集准确度：{evaluate(model,x_test,y_test)*100:.2f}")

