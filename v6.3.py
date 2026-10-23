"""
版本 6：把"特征提取器"换成卷积网络（CNN）
==========================================
前 5 版都是全连接网络，准确率卡在 98% 左右上不去了。原因不在训练，
而在**结构**：全连接层把图片摊平成 784 个互不相关的数字，
丢掉了"哪些像素是相邻的"这个极其重要的信息。

V6 只改一个地方——把 MLP 的 features 换成两层卷积。
训练代码、数据代码、评估代码，一行都不用动。
这就是 V5 把网络拆成 features/classifier 的意义。

结果：98.02% -> 99% 以上。

运行：python v6_cnn.py        （约 90 秒）
"""

import copy

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# ══════════════ 唯一的改动：网络结构 ══════════════
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
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # 输入 1 通道(灰度) -> 32 个特征图
            nn.ReLU(),
            nn.MaxPool2d(2),                            # 28x28 -> 14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 32 通道 -> 64 通道
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 14x14 -> 7x7
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),                   # 3136 -> 128
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)              # (N,64,7,7)
        x = x.view(x.size(0), -1)         # 展平：(N,64,7,7) -> (N,3136)
        x = self.classifier(x)            # (N,10)
        return x


def build_model():
    return CNN()


# ══════════════ 以下是 V5 的原样代码，只改了优化器那一行 ══════════════
def get_loaders(batch_size=64):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    full = datasets.MNIST("./data", train=True, download=True, transform=tf)
    test = datasets.MNIST("./data", train=False, download=True, transform=tf)

    train_set, val_set = random_split(
        full, [54000, 6000],
        generator=torch.Generator().manual_seed(42),
    )
    return (DataLoader(train_set, batch_size=batch_size, shuffle=True),
            DataLoader(val_set, batch_size=batch_size, shuffle=False),
            DataLoader(test, batch_size=batch_size, shuffle=False))


def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            pred = model(images).argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    return correct / total


def train(model, train_loader, val_loader, lr=0.003, epochs=4):
    loss_fn = nn.CrossEntropyLoss()
    # 这里从 SGD 换成 Adam：卷积网络的参数更多、梯度尺度差别更大，
    # Adam 会为每个参数单独调整步长，比手调 SGD 的学习率省事得多。
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc, best_state = 0.0, None
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()

        scheduler.step()
        val_acc = evaluate(model, val_loader)
        print(f"第 {epoch + 1} 轮  lr={scheduler.get_last_lr()[0]:.4f}  验证acc={val_acc * 100:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    print(f"最佳验证集准确率：{best_acc * 100:.2f}%")


if __name__ == "__main__":
    torch.manual_seed(0)
    train_loader, val_loader, test_loader = get_loaders()
    model = build_model()
    print(f"模型参数量：{sum(p.numel() for p in model.parameters()):,}\n")

    train(model, train_loader, val_loader)
    print(f"测试集准确率：{evaluate(model, test_loader) * 100:.2f}%")
