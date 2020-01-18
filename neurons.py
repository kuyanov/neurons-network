import os
import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as ff
import torch.optim as optim
import torchvision.transforms as transforms


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.pool(ff.relu(self.conv1(x)))
        x = self.pool(ff.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = ff.relu(self.fc1(x))
        x = ff.relu(self.fc2(x))
        x = self.fc3(x)
        return ff.log_softmax(x)


net = Net()
optimizer = optim.SGD(net.parameters(), lr=0.00001, momentum=0.6)
criterion = nn.CrossEntropyLoss()

if os.path.exists('neurons.pt'):
    checkpoint = torch.load('neurons.pt')
    net.load_state_dict(checkpoint['model_state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    net.train()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=4, shuffle=True, num_workers=2
)
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=4, shuffle=False, num_workers=2
)

for epoch in range(1):
    cur_loss = 0.0
    for batch_idx, (inputs, labels) in enumerate(trainloader, 0):
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        cur_loss += loss.item()
        if batch_idx % 1000 == 999:
            print('epoch:', epoch + 1, 'idx:', batch_idx + 1, 'loss:', cur_loss / 1000)
            cur_loss = 0.0

print('Finished training')

torch.save({
    'model_state_dict': net.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()
}, 'neurons.pt')

correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in testloader:
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of network is {} %'.format(100 * correct / total))

