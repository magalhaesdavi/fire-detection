import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
        transforms.Resize((50, 50)),
        transforms.ToTensor(),
        transforms.Normalize((0.6558, 0.4875, 0.2858), (0.3469, 0.3010, 0.2526))
        ])

data_path = '../dataset/'

train_set = torchvision.datasets.ImageFolder(root='../split_dataset/train', transform=transform)
val_set = torchvision.datasets.ImageFolder(root='../split_dataset/val', transform=transform)
test_set = torchvision.datasets.ImageFolder(root='../split_dataset/test', transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                           batch_size=4,
                                           shuffle=True)


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    print(inp.shape)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


class_names = train_set.classes
inputs, classes = next(iter(train_loader))
# Make a grid from batch
out = torchvision.utils.make_grid(inputs)
imshow(out, title=[class_names[x] for x in classes])

# temp_img, temp_lab = train_set[2]
# plt.imshow(temp_img.numpy().transpose((1, 2, 0)))
# plt.title(temp_lab)
# plt.axis('off')
# plt.show()
