#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 04:02:43 2018

@author: Avinash Kadimisetty
"""

# =============================================================================
# This function loads the dataset from Pytorch datasets and returns a 
# a dataloader which can be used while training and testing
# =============================================================================
def generate_loaders(data_directory, bs):
    
    # List out all the data augmentation techniques needed using transforms.
    data_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                     transforms.ColorJitter(),
                                     transforms.toTensor()])
    
    # Download the datasets from PyTorch
    train_dataset = torchvision.datasets.MNIST(root = ".",
                                               train = True,
                                               download = True,
                                               transforms = data_transforms)
    test_dataset = torchvision.datasets.MNIST(root = ".",
                                              train = False,
                                              download = False,
                                              transforms = transforms.toTensor())
    
    # Create loaders
    train_loader = torch.utils.data.DataLoader(data = train_dataset,
                                               batch_size = bs,
                                               shuffle = True)
    test_loader = torch.utils.data.DataLoader(data = test_dataset,
                                              batch_size = bs,
                                              shuffle = False)
    return train_loader, test_loader