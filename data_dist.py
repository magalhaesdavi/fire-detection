import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_classes = 2
batch_size = 16
learning_rate = 0.001

transform = transforms.Compose([
        transforms.Resize((50, 50)),
        transforms.ToTensor()
        # transforms.Normalize((0.6083, 0.6083, 0.6083), (1.1406, 1.1406, 1.1406))
        ])

# data_set = torchvision.datasets.ImageFolder(root=data_path, transform=transform)
# train_set = torchvision.datasets.ImageFolder(root='../dataset/train', transform=transform)
train_set = torchvision.datasets.ImageFolder(root='../fire-dataset/train/trainloader/',
                                             transform=transform)
# val_set = torchvision.datasets.ImageFolder(root='../dataset/validation')
# test_set = torchvision.datasets.ImageFolder(root='../dataset/test')

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                           batch_size=batch_size,
                                           num_workers=4,
                                           shuffle=False)

# val_loader = torch.utils.data.DataLoader(dataset=val_set,
#                                          batch_size=batch_size,
#                                          shuffle=False)
#
# test_loader = torch.utils.data.DataLoader(dataset=test_set,
#                                           batch_size=batch_size,
#                                           shuffle=False)
if __name__ == '__main__':
    mean = torch.zeros(3)
    std = torch.zeros(3)

    for i, (images, labels) in enumerate(train_loader):
        data = images
        # if i % 10000 == 0:
        #     print(i)
        data = data[0].squeeze(0)
        if i == 0: size = data.size(1) * data.size(2)
        mean += data.sum((1, 2)) / size

    mean /= len(train_loader)
    print(mean)
    mean = mean.unsqueeze(1).unsqueeze(2)

    for i, (images, labels) in enumerate(train_loader):
        data = images
        # if i % 10000 == 0:
        #     print(i)
        data = data[0].squeeze(0)
        std += ((data - mean) ** 2).sum((1, 2)) / size

    std /= len(train_loader)
    std = std.sqrt()
    print(std)
