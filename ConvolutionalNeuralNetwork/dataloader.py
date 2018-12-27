# @Author: Avinash Kadimisetty <avinash>
# @Date:   2018-12-27T12:30:53-06:00
# @Project: ConvolutionalNeuralNetwork
# @Filename: dataloader.py
# @Last modified by:   avinash
# @Last modified time: 2018-12-27T12:53:15-06:00


# =============================================================================
# This function loads the dataset from Pytorch datasets and returns a
# a dataloader which can be used while training and testing
# =============================================================================

import torch
import torchvision.transforms as transforms
import torchvision


def generate_loaders(data_directory, batchsize):

    # List out all the data augmentation techniques needed using transforms.
    train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                           transforms.ColorJitter(),
                                           transforms.ToTensor()])

    test_transforms = transforms.Compose([transforms.ToTensor()])

    # Download the datasets from PyTorch
    train_dataset = torchvision.datasets.CIFAR10(root=data_directory,
                                                 train=True,
                                                 download=True,
                                                 transform=train_transforms)
    test_dataset = torchvision.datasets.CIFAR10(root=data_directory,
                                                train=False,
                                                download=True,
                                                transform=test_transforms)

    # Create loaders
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batchsize,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batchsize,
                                              shuffle=False)
    return train_loader, test_loader
