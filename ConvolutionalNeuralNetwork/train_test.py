# @Author: Avinash Kadimisetty <avinash>
# @Date:   2018-12-27T12:31:33-06:00
# @Project: ConvolutionalNeuralNetwork
# @Filename: train.py
# @Last modified by:   avinash
# @Last modified time: 2018-12-27T12:48:40-06:00

# =============================================================================
# This file contains functions to train and test the model
# =============================================================================

import torch
from torch.autograd import Variable
import torch.optim.optimizer
import numpy as np
import torch.nn.functional as F

from dataloader import generate_loaders
from model_architecture import CNN

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

num_classes = 10
batch_size = 100

train_loader, test_loader = generate_loaders(".", batch_size)
model = CNN(num_classes)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train(trainloader, model, optimizer):
    model.train()

    train_loss = []
    train_acc = []

    for i, (features, labels) in enumerate(trainloader):
        data = Variable(features).to(device)
        target = Variable(labels).to(device)
        optimizer.zero_grad()

        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(output.data, 1)
        correct = (predicted == labels).sum().item()
        accuracy = format(100 * correct / batch_size)

        train_acc.append(accuracy)
        train_loss.append(loss.item())

    loss_train = np.mean(train_loss)
    acc = np.mean(train_acc)
    return loss_train, acc


def test(testloader, model):
    model.eval()

    with torch.no_grad():
        test_loss = []
        test_acc = []

        for i, (features, labels) in enumerate(testloader):
            data = Variable(features).to(device)
            target = Variable(labels).to(device)
            optimizer.zero_grad()

            output = model(data)
            loss = F.nll_loss(output, target)

            _, predicted = torch.max(output.data, 1)
            correct = (predicted == labels).sum().item()
            accuracy = format(100 * correct / batch_size)
            test_acc.append(accuracy)
            test_loss.append(loss.item())

        loss_test = np.mean(test_loss)
        acc = np.mean(test_acc)
        return loss_test, acc


epoch_train_loss = []
epoch_test_loss = []
epoch_train_accuracy = []
epoch_test_accuracy = []
num_epochs = 2
for epoch in range(num_epochs):
    train_loss, train_acc = train(train_loader, model, optimizer)
    print('[Epoch: %3d/%3d][Train Loss: %5.5f][Train Acc: %5.5f]' %
          (epoch, num_epochs, train_loss, train_acc))
    test_loss, test_acc = test(test_loader, model)
    print('[Epoch: %3d/%3d][Test Loss: %5.5f][Test Acc: %5.5f]' %
          (epoch, num_epochs, test_loss, test_acc))
    epoch_train_loss.append(train_loss)
    epoch_train_accuracy.append(train_acc)
    epoch_test_loss.append(test_loss)
    epoch_test_accuracy.append(test_acc)
