#网络写成类
#features  = 特征提取器（把 784 个像素变成 128 个特征）
   # classifier = 分类器（把 128 个特征变成 10 个分数）
import copy
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader,random_split
from torchvision import datasets, transforms

#网络：一串层到一个类
class MLP(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 10),
        ) # 输出 10 个"原始打分"（logits），不接 Softmax，
        # 因为 CrossEntropyLoss 内部已经合并了 Softmax
    def forward(self, x):
        """数据怎么流过这个网络，全写在这里。

               注意：调用时写 model(x)，不要写 model.forward(x)。
               因为 model(x) 会先走父类的 __call__，它除了调 forward，
               还会处理钩子(hook)等内部机制，是 PyTorch 的约定。
               """
        x = self.features(x)#先提特征
        x= self.classifier(x)#分类
        return x

def build_model():
    return MLP()

def get_loaders(batch_size=64):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    full = datasets.MNIST(root='./data', train=True, transform=tf, download=True)
    test = datasets.MNIST(root='./data', train=False, transform=tf, download=True)

    train_set, val_set = random_split(
        full,[54000,6000],
        generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(
        train_set,batch_size=batch_size,shuffle=True)
    val_loader = DataLoader(
        val_set,batch_size=batch_size,shuffle=False)
    test_loader = DataLoader(
        test,batch_size=batch_size,shuffle=False)
    return train_loader,val_loader,test_loader

def evaluate(loader,model):
    model.eval()
    correct,total = 0,0
    with torch.no_grad():
        for images,labels in loader:
            pred = model(images).argmax(dim=1)
            correct += (pred==labels).sum().item()
            total += labels.size(0)
    return correct / total

def train(model,train_loader,val_loader,lr=0.3,epochs=8):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD  (model.parameters(),lr=lr)
    scheduler = CosineAnnealingLR(optimizer,T_max=epochs)

    best_acc , best_state = 0.0 , None
    for epoch in range(epochs):
        model.train()
        for images,labels in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(images),labels)
            loss.backward()
            optimizer.step()

        scheduler.step()
        val_acc = evaluate(val_loader,model)
        print(f"第{epoch+1}轮  lr={scheduler.get_last_lr()[0]:.4f}  验证acc={val_acc * 100:.2f}%")
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    print(f"最佳的验证集准确值；{best_acc*100:.2f}%")

if __name__ == '__main__':
    torch.manual_seed(42)
    train_loader,val_loader,test_loader = get_loaders()
    model = build_model()
    # 顺手看一眼"这个类到底有多少参数"，这是 V4 里做不到的
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量：{n_params:,}\n")

    train(model,train_loader,val_loader)
    print(f"测试集的准确率：{evaluate(test_loader,model) * 100:.2f}%")