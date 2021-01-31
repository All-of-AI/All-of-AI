# 使用pytorch创建神经网络，并对MNIST手写数字进行识别

import torch
import torchvision
import torch.utils.data.dataloader as Data
import torch.nn.functional as F
from torch.autograd import Variable

# 获取MNIST数据
train_data = torchvision.datasets.MNIST(
    './data', train=True, transform=torchvision.transforms.ToTensor(), download=False)

test_data = torchvision.datasets.MNIST(
    './data', train=False, transform=torchvision.transforms.ToTensor(), download=False)

print("train_data:", train_data.train_data.size())
print("train_labels:", train_data.train_labels.size())
print("test_data:", test_data.test_data.size())

# 数据加载器
train_loader = Data.DataLoader(dataset=train_data, batch_size=64, shuffle=True)
test_loader = Data.DataLoader(dataset=test_data, batch_size=64)


# 网络结构
class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 500)
        self.fc2 = torch.nn.Linear(500, 256)
        self.fc3 = torch.nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = NeuralNetwork()
print(model)

# 定义优化器和损失函数
optimizer = torch.optim.Adam(model.parameters())
loss_func = torch.nn.CrossEntropyLoss()

for epoch in range(10):
    print('epoch {}'.format(epoch + 1))

    # 训练
    train_loss = 0.
    train_acc = 0.

    for batch_x, batch_y in train_loader:
        batch_x, batch_y = Variable(batch_x), Variable(batch_y)

        out = model(batch_x)
        loss = loss_func(out, batch_y)
        train_loss += loss.item()
        pred = torch.max(out, 1)[1]
        train_correct = (pred == batch_y).sum()
        train_acc += train_correct.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(
        train_data)), train_acc / (len(train_data))))

    # 测试
    model.eval()
    eval_loss = 0.
    eval_acc = 0.
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = Variable(batch_x, volatile=True), Variable(batch_y, volatile=True)

        out = model(batch_x)
        loss = loss_func(out, batch_y)
        eval_loss += loss.item()
        pred = torch.max(out, 1)[1]
        num_correct = (pred == batch_y).sum()
        eval_acc += num_correct.item()

    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(test_data)), eval_acc / (len(test_data))))
