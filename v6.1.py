#版本 6：把"特征提取器"换成卷积网络（CNN
import torch
import copy
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader,random_split
from torchvision import datasets,transforms

class CNN(nn.Module):
    """卷积网络：保留图片的空间结构。

        形状怎么变（面试高频手算题）：
            输入            (N,  1, 28, 28)
            Conv 3x3 pad=1  (N, 32, 28, 28)   通道 1->32，尺寸不变（pad=1 保住）
            MaxPool 2x2     (N, 32, 14, 14)   边长减半
            Conv 3x3 pad=1  (N, 64, 14, 14)   通道 32->64，尺寸不变
            MaxPool 2x2     (N, 64,  7,  7)   边长再减半
            view 展平       (N, 3136)         64 * 7 * 7 = 3136
            Linear          (N, 10)

        卷积比全连接强在哪（三个理由）：
          ① 局部连接：一个 3x3 卷积核只看一小块，先学边缘、再组合成笔画，符合直觉
          ② 权值共享：同一个卷积核在全图滑动，参数被复用。第一层只要 320 个参数，
                      而全连接第一层要 784*128+128 = 100,480 个
          ③ 平移不变：数字写在左上角还是右下角，卷积都能认出同一个模式
        """
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=64,kernel_size=3,stride=1),
            # 输入 1 通道(灰度) -> 32 个特征图
            nn.ReLU(),
            nn.MaxPool2d(2),# 28x28 -> 14x14
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1),
            # 32 通道 -> 64 通道
            nn.ReLU(),
            nn.MaxPool2d(2),)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=7*7*64,out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128,out_features=10),
        )
    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)
        return x

def build_model():
    return CNN()

def get_loader(batch_size=64):
    tf = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),])
    full = datasets.MNIST(root='./data',train=True,transform=tf,download=True)
    test = datasets.MNIST(root='./data',train=False,transform=tf,download=True)

    train_set,val_set = random_split(
        full,[54000,6000],
        generator=torch.Generator().manual_seed(42))
    return (DataLoader(train_set,batch_size=batch_size,shuffle=True),
            DataLoader(val_set,batch_size=batch_size,shuffle=False),
            DataLoader(test,batch_size=batch_size,shuffle=False))

def evaluate(loader,model):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            pred = model(images).argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    return correct/total

def train(model,train_loader,val_loader,lr=0.003,epochs=4):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    # 这里从 SGD 换成 Adam：卷积网络的参数更多、梯度尺度差别更大，
    # Adam 会为每个参数单独调整步长，比手调 SGD 的学习率省事得多。
    scheduler = CosineAnnealingLR(optimizer,T_max=epochs)

    best_acc,best_state = 0.0,None

    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(images),labels)
            loss.backward()
            optimizer.step()

        scheduler.step()
        val_acc = evaluate(val_loader,model)
        print(f"第{epoch+1}轮   lr={scheduler.get_lr()[0]:.4f}"
              f"验证acc= {val_acc*100:.4f}%")
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    print(f"最佳验证集准确率：{best_acc*100:.4f}%")

if __name__ == '__main__':
    torch.manual_seed(42)
    train_loader = get_loader(batch_size=64)
    val_loader = get_loader(batch_size=64)
    test_loader = get_loader(batch_size=64)
    model = build_model()
    print(f"模型参数：{sum(p.numel() for p in model.parameters()):,}\n")

    train(model,train_loader,val_loader)
    print(f"测试集准确率：{evaluate(test_loader,model)*100:.4f}%")
