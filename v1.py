import torch
import torch.nn as nn
from mpmath.identification import transforms
from torchvision import transforms, datasets

torch.manual_seed(42)

train_data = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_data = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)

x = torch.stack([image for image,_ in train_data]).reshape(-1,784)
y = torch.tensor([label for _, label in train_data])

x_test = torch.stack([image for image,_ in test_data]).reshape(-1,784)
y_test = torch.tensor([label for _,label in test_data])

model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1.0)

for epoch in range(100):
    optimizer.zero_grad()
    output = model(x)
    loss = loss_fn(output, y)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}: Loss {loss.item():.4f}')

with torch.no_grad():
    pre = model(x_test).argmax(dim=1)
    accuracy = (pre == y_test).float().mean()
print(f"训练的测试概率: {accuracy.item()*100:.3f}%")




