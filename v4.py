import copy
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader,random_split
#       Dataset    -> 一条一条地取样本
#    DataLoader -> 自动分批 + 自动打乱
#   random_split -> 自动切训练/验证集

#数据处理
def get_loaders(batch_size=64):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    full = datasets.MNIST(
        root='./data',train=True,transform=tf,download=True)
    test = datasets.MNIST(
        root='./data',train=False,transform=tf,download=True)

    #60000张切成54000训练和6000验证
    train_set,val_set = random_split(
        full,[54000,6000],
        generator=torch.Generator().manual_seed(42),
        #generator用来固定随机种子
    )
    train_loader = DataLoader(
        train_set,batch_size=batch_size,shuffle=True)
    val_loader = DataLoader(
        val_set,batch_size=batch_size,shuffle=False)
    test_loader = DataLoader(
        test,batch_size=batch_size,shuffle=False)
    return train_loader,val_loader,test_loader
"""返回 (训练loader, 验证loader, 测试loader)。

    三个要点：
      ① transform 直接传给 datasets.MNIST，它在"取出一张图"时自动套用，
         不用我们再手动堆叠、reshape。
      ② random_split 负责切分。用 generator 固定随机种子，
         保证每次运行切出来的训练/验证集完全一样（否则结果会抖）。
      ③ DataLoader 负责分批和打乱：
           shuffle=True  每个 epoch 自动换顺序（训练集必须打乱）
           shuffle=False 不打乱（验证/测试不需要，而且顺序固定方便按索引取图）
    """
def build_model():
    return nn.Sequential(
        nn.Flatten(),          # (N,1,28,28) -> (N,784)，自动压平，省掉 reshape
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )
#评估（吃loader）
def evaluate(model,loader):
    model.eval()# 切到评估模式（虽然本网络没有 Dropout，但养成习惯）
    correct ,total = 0, 0       #correct计数几个对，total计数总共有几个数据
    with torch.no_grad():
        for images,labels in loader:    # ← 这里自动就是一批一批的
            pred = model(images).argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)     #防止最后一轮的label数量不足
    return correct / total
#训练
def train(model,train_loader,val_loader,lr=0.4,epochs=8):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=lr)
    scheduler = CosineAnnealingLR(optimizer,T_max=epochs)   #T_max是余弦退火的一个完整周长

    best_acc,best_state = 0.0,None
    for epoch in range(epochs):
        model.train()
        for images,labels in train_loader:

            optimizer.zero_grad()
            loss = loss_fn(model(images),labels)
            loss.backward()
            optimizer.step()

        scheduler.step()
        val_acc = evaluate(model,val_loader)
        print(f"{epoch+1}轮  lr={scheduler.get_last_lr()[0]:.4f}  "
              f"验证acc={val_acc * 100:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

        model.load_state_dict(best_state)
        print(f"最佳验证集准确率：{best_acc * 100:.2f}%")

if __name__ == "__main__":
    torch.manual_seed(42)
    train_loader,val_loader,test_loader = get_loaders()
    model =build_model()
    train(model,train_loader,val_loader)

    print(f"测试准确率：{evaluate(model,test_loader)*100:.2f}%")







