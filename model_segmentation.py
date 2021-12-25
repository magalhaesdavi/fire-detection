import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image, ImageDraw
from models import coatnet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
nb_classes = 2

# transforms_nasnetlarge = transforms.Compose([
#         transforms.Resize((331, 331)),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#         ])

transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

# transforms_senet = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#         ])
#
# transforms_nasnet = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # pytorch models:(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
#         ])

# model_ff = nasnetamobile.NASNetAMobile(num_classes=nb_classes)
# model_ff.load_state_dict(torch.load('weights/nasnet_ff.ckpt', map_location=device))
# model_ff.eval()
# model_ff = model_ff.to(device)


# model_name = 'senet154'  # could be fbresnet152 or inceptionresnetv2
# model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')

# fine tuning
# dim_feats = model.last_linear.in_features
# model.last_linear = nn.Linear(dim_feats, nb_classes)
# model.load_state_dict(torch.load('weights/senet154.ckpt'))
# model = model.to(device)
# model.eval()

ff_model = coatnet.coatnet_4()
ff_model.load_state_dict(torch.load('weights/ffcoatnet_4.ckpt'))
ff_model = ff_model.to(device)
ff_model.eval()

model = coatnet.coatnet_4()
model.load_state_dict(torch.load('weights/coatnet_4.ckpt'))
model = model.to(device)
model.eval()

(w_width, w_height) = (50, 50)
step_size = 25

input_path = 'dataset/test/img/'

if os.path.isdir(input_path):
    lst_img = [os.path.join(input_path, file)
               for file in os.listdir(input_path)]
else:
    if os.path.isfile(input_path):
        lst_img = [input_path]
    else:
        raise Exception("Invalid path")

for im in lst_img:
    print('\t|____Image processing: ', im)
    # image = cv2.imread(im)
    # image = np.array(Image.open(im))
    # img = Image.new('RGB', (image.shape[1], image.shape[0]), color='black')
    # d = ImageDraw.Draw(img)
    # for x in range(0, image.shape[1], step_size):
    #     for y in range(0, image.shape[0], step_size):
    #         window = image[y:y + w_height, x:x + w_width]
    #         frame = window.copy()
    #         frame = Image.fromarray(frame)
    #         frame = transform(frame).float()
    #         frame = frame.unsqueeze(0)
    #         frame = frame.to(device)
    #         output = model(frame)  # [0]
    #         _, prediction = torch.max(output.data, 1)
    #
    #         if prediction == 0:
    #             d.rectangle(((x, y), (x + w_width - 1, y + w_height - 1)), fill=(255, 255, 255))

    image = np.array(Image.open(im))
    img = Image.new('RGB', (image.shape[1], image.shape[0]), color='black')

    frame = image
    frame = Image.fromarray(frame)
    frame = transform(frame).float()
    frame = frame.unsqueeze(0)
    frame = frame.to(device)
    output = ff_model(frame)
    # print(output)
    softmax = nn.Softmax()
    output = softmax(output)
    # print(output)
    _, prediction = torch.max(output.data, 1)
    # print(prediction)

    if prediction == 0 or (prediction == 1 and _ < 0.98):
        d = ImageDraw.Draw(img)
        for x in range(0, image.shape[1], step_size):
            for y in range(0, image.shape[0], step_size):
                window = image[y:y + w_height, x:x + w_width]
                frame = window.copy()
                frame = Image.fromarray(frame)
                frame = transform(frame).float()
                frame = frame.unsqueeze(0)
                frame = frame.to(device)
                output = model(frame)
                _, prediction = torch.max(output.data, 1)

                if prediction == 0:
                    d.rectangle(((x, y), (x + w_width - 1, y + w_height - 1)), fill=(255, 255, 255))

    output_path = 'output/model/coatnet_4_preclass0.98/' + im[im.rfind("/") + 1:]
    img.save(output_path)
