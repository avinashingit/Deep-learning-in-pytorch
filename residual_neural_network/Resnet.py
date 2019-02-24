import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim.optimizer
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms

torch.manual_seed(89)
torch.cuda.manual_seed(89)
np.random.seed(89)

batch_size = 128

data_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                     transforms.RandomCrop(padding=4, size = 28),
                                     transforms.ToTensor()])

train_dataset = torchvision.datasets.CIFAR100(root='./',
                                           train=True,
                                           transform=data_transform,
                                           download=True)

test_dataset = torchvision.datasets.CIFAR100(root='./',
                                          train=False,
                                          transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

class BasicBlock(nn.Module):

    def __init__(self, in_c, out_c, stride, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.downsample = downsample

    def forward(self, X):
        residual = X
        out = self.conv1(X)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(X)
        out += residual
        return out

class ResidualNetwork(nn.Module):

    def __init__(self, basicBlock):
        super(ResidualNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout2d(p=0.2)
        self.bb1 = self.addBasicBlock(basicBlock, 32, 32, 2, 1)
        self.bb2 = self.addBasicBlock(basicBlock, 32, 64, 4, 2)
        self.bb3 = self.addBasicBlock(basicBlock, 64, 128, 4, 2)
        self.bb4 = self.addBasicBlock(basicBlock, 128, 256, 2, 2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(in_features=256*4, out_features=100)

    def addBasicBlock(self, basicBlock, channels_in, channels_out, num_blocks,
                      stride):
        blocks = []
        downsample = None
        if channels_in != channels_out:
            downsample = nn.Sequential(nn.Conv2d(in_channels=channels_in,
                                                 out_channels=channels_out,
                                                 padding=0, kernel_size=1,
                                                 stride=stride),
                                      nn.BatchNorm2d(channels_out))
        blocks.append(basicBlock(channels_in, channels_out, stride, downsample))
        for b in range(1, num_blocks):
            blocks.append(basicBlock(channels_out, channels_out, 1))
        return nn.Sequential(*blocks)

    def forward(self, X):
        out = self.conv1(X)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.bb1(out)
        out = self.bb2(out)
        out = self.bb3(out)
        out = self.bb4(out)
        out = self.pool1(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = F.log_softmax(out, dim = 1)
        return out

model = ResidualNetwork(BasicBlock)
model.cuda()

def train(trainloader, model, optimizer, criterion):
    model.train()

    train_loss = []
    correct = 0
    total = 0
    train_acc = []

    for i, (features, labels) in enumerate(trainloader):
        data, target = Variable(features).cuda(), Variable(labels).cuda()
        optimizer.zero_grad()

        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        prediction = output.data.max(1)[1]
        accuracy = (float(prediction.eq(target.data).sum())/float(batch_size))*100.0
        train_acc.append(accuracy)
        train_loss.append(loss.data[0])

    loss_train = np.mean(train_loss)
    acc = np.mean(train_acc)
    return loss_train, acc

def test(testloader, model, criterion):
    model.eval()

    test_loss = []
    test_acc = []
    correct = 0
    total = 0

    for i, (features, labels) in enumerate(testloader):
        data, target = Variable(features).cuda(), Variable(labels).cuda()

        output = model(data)
        loss = F.nll_loss(output, target)

        prediction = output.data.max(1)[1]
        accuracy = (float(prediction.eq(target.data).sum())/float(batch_size))*100.0
        test_acc.append(accuracy)
        test_loss.append(loss.data[0])

    loss_test = np.mean(test_loss)
    acc = np.mean(test_acc)
    return loss_test, acc

optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
criterion = nn.CrossEntropyLoss()
epoch_train_loss = []
epoch_test_loss = []
epoch_train_accuracy = []
epoch_test_accuracy = []
num_epochs = 94
for epoch in range(num_epochs):
        train_loss, train_acc = train(train_loader, model, optimizer, criterion)
        print('[Epoch: %3d/%3d][Train Loss: %5.5f][Train Acc: %5.5f]' %
              (epoch, num_epochs, train_loss, train_acc))
        test_loss, test_acc = test(test_loader, model, criterion)
        print('[Epoch: %3d/%3d][Test Loss: %5.5f][Test Acc: %5.5f]' %
              (epoch, num_epochs, test_loss, test_acc))
        # torch.save(model.state_dict(), './' + str(epoch+1)+ '.pth')
        epoch_train_loss.append(train_loss)
        epoch_train_accuracy.append(train_acc)
        epoch_test_loss.append(test_loss)
        epoch_test_accuracy.append(test_acc)

# import matplotlib.pyplot as plt
# plt.plot(range(94), epoch_train_accuracy, label = "Train Accuracy")
# plt.plot(range(94), epoch_test_accuracy, label = "Test Accuracy")
# plt.title("Epoch vs Accuracy")
# plt.xlabel("Epoch Number")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.show()
#
# plt.plot(range(94), epoch_train_loss, label = "Train Loss")
# plt.plot(range(94), epoch_test_loss, label = "Test Loss")
# plt.title("Epoch vs Loss")
# plt.xlabel("Epoch Number")
# plt.ylabel("Loss")
# plt.legend()
# plt.show()
