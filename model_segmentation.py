import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image, ImageDraw
from models import coatnet
from sklearn import metrics
import pandas as pd
import seaborn as sns

# TODO: threshold no modelo de superpixel para a melhor combinação até agora e para as últimas combinações feitas
# TODO: modelo de sp novo no fullframe antigo
# TODO: testar diferentes stepsizes

def preclass_segmentation(device):
    nb_classes = 2
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    ff_model = coatnet.coatnet_4(one_fc=True)
    ff_model.load_state_dict(torch.load('weights/ffcoatnetv2.ckpt'))
    ff_model = ff_model.to(device)
    ff_model.eval()

    input_path = 'datasets/test/img/'

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

        image = np.array(Image.open(im))
        width = image.shape[1]
        height = image.shape[0]

        frame = image
        frame = Image.fromarray(frame)
        frame = transform(frame).float()
        frame = frame.unsqueeze(0)
        frame = frame.to(device)
        output = ff_model(frame)
        # print(output)
        softmax = nn.Softmax(dim=1)
        output = softmax(output)
        # print(output)
        _, prediction = torch.max(output.data, 1)
        # print(prediction)

        flag = False
        if prediction == 1 and _ < 0.91:
            prediction = 0
            flag = True
        if prediction == 0 and _ < 0.6 and not flag:
            prediction = 1

        if prediction == 0:
            img = Image.new('RGB', (width, height), color=(255, 255, 255))
            output_path = 'output/model/preclass/' + im[im.rfind("/") + 1:]
            img.save(output_path)
        else:
            img = Image.new('RGB', (width, height), color='black')
            output_path = 'output/model/preclass/' + im[im.rfind("/") + 1:]
            img.save(output_path)


def main(device):
    # transforms_nasnetlarge = transforms.Compose([
    #         transforms.Resize((331, 331)),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    #         ])
    softmax = nn.Softmax(dim=1)
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

    ff_model = coatnet.coatnet_4(one_fc=True)
    # ff_model.load_state_dict(torch.load('weights/ffcoatnet472.ckpt'))
    ff_model.load_state_dict(torch.load('weights/ffcoatnetv2.ckpt'))
    ff_model = ff_model.to(device)
    ff_model.eval()

    model = coatnet.coatnet_4(one_fc=True)
    model.load_state_dict(torch.load('weights/coatnet_4.ckpt'))
    # model.load_state_dict(torch.load('weights/spcoatnet1.ckpt'))
    model = model.to(device)
    model.eval()

    (w_width, w_height) = (50, 50)
    step_size = 25

    input_path = 'datasets/test/img/'

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
        output = softmax(output)
        # print(output)
        prob, prediction = torch.max(output.data, 1)
        # print(prediction)
        # print(prob.item())

        # 0.8 and 0.7
        flag = False
        if prediction == 0 and prob < 0.8:
            prediction = 1
            flag = True
        if prediction == 1 and prob < 0.7 and not flag:
            prediction = 0

        # if prediction == 0 or (prediction == 1 and _ < 0.98):
        if prediction == 0:
            # img = Image.new('RGB', (image.shape[1], image.shape[0]), color='white')
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
                    output = softmax(output)
                    _, prediction = torch.max(output.data, 1)

                    # if prediction == 1 and _ < 0.7:
                    #     prediction = 0
                    #
                    if prediction == 0:
                        d.rectangle(((x, y), (x + w_width - 1, y + w_height - 1)), fill=(255, 255, 255))

        output_path = 'output/model/preclass/' + im[im.rfind("/") + 1:]
        img.save(output_path)

def compute_auc(device):
    transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

    ff_model = coatnet.coatnet_4(one_fc=True)
    ff_model.load_state_dict(torch.load('weights/ffcoatnet_4.ckpt'))
    ff_model = ff_model.to(device)
    ff_model.eval()

    model = coatnet.coatnet_4(one_fc=True)
    model.load_state_dict(torch.load('weights/coatnet_4.ckpt'))
    model = model.to(device)
    model.eval()

    (w_width, w_height) = (50, 50)
    step_size = 25

    input_path = 'datasets/test/img/'
    gt_path = 'datasets/test/gt/'

    if os.path.isdir(input_path):
        lst_img = [os.path.join(input_path, file)
                   for file in os.listdir(input_path)]
    else:
        if os.path.isfile(input_path):
            lst_img = [input_path]
        else:
            raise Exception("Invalid path")

    flattened_y_score = []
    flattened_y_score = np.array(flattened_y_score)
    flattened_y_true = []
    flattened_y_true = np.array(flattened_y_true)

    for im in lst_img:
        print('\t|____Image processing: ', im)

        # y_true_path = cv2.imread(gt_path + (im[im.rfind("/") + 1:]).replace('.', '_gt.'))
        y_true = np.array(Image.open(gt_path + (im[im.rfind("/") + 1:]).replace('.', '_gt.')))
        y_true = y_true[:, :, 0]
        y_true = y_true / 255
        y_true = y_true.flatten()
        y_true = 1 - y_true
        # print(y_true)
        y_true = y_true.astype(int)
        # print(y_true)
        flattened_y_true = np.concatenate((flattened_y_true, y_true))

        image = np.array(Image.open(im))

        y_score = np.ones((image.shape[0], image.shape[1]))
        # img = Image.new('RGB', (image.shape[1], image.shape[0]), color='black')

        frame = image
        frame = Image.fromarray(frame)
        frame = transform(frame).float()
        frame = frame.unsqueeze(0)
        frame = frame.to(device)
        output = ff_model(frame)
        # print(output)
        softmax = nn.Softmax(dim=1)
        output = softmax(output)
        # print(output[0][0].item())
        fire_prob = output[0][0].item()
        prob, prediction = torch.max(output.data, 1)
        # print(prediction)
        # print(prob.item())

        flag = False
        if prediction == 1 and prob < 0.98:
            prediction = 0
        #     flag = True
        # if prediction == 0 and _ < 0.6 and not flag:
        #     prediction = 1

        if prediction == 0:
            for x in range(0, image.shape[1], step_size):
                for y in range(0, image.shape[0], step_size):
                    window = image[y:y + w_height, x:x + w_width]
                    frame = window.copy()
                    frame = Image.fromarray(frame)
                    frame = transform(frame).float()
                    frame = frame.unsqueeze(0)
                    frame = frame.to(device)
                    output = model(frame)
                    output = softmax(output)
                    fire_prob = output[0][0].item()
                    # print(fire_prob)
                    # prob, prediction = torch.max(output.data, 1)
                    # y_score[x:x + w_width, y:y + w_height] = prob.item()
                    y_score[x:x + w_width, y:y + w_height] = fire_prob

            flattened_y_score = np.concatenate((flattened_y_score, y_score.flatten()))
        else:
            # y_score *= prob.item()
            y_score *= fire_prob
            flattened_y_score = np.concatenate((flattened_y_score, y_score.flatten()))

    return flattened_y_true, flattened_y_score

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main(device)
    # preclass_segmentation(device)

    # y_true, y_score = compute_auc(device)
    # fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    # auc = metrics.roc_auc_score(y_true, y_score)

    # d = {'FPR': fpr, 'TPR': tpr}
    # df = pd.DataFrame(d)
    #
    # roc_curve = sns.relplot(
    #     x='FPR',
    #     y='TPR',
    #     data=df,
    #     # hue='algorithm',
    #     kind="line",
    #     lw=3,
    #     # style="algorithm",
    #     markers=True,
    #     dashes=False
    # )
    # roc_curve.set_axis_labels("FPR", "TPR")
    # roc_curve.fig.suptitle('ROC curve')
    # roc_curve.fig.title_fontsize = 18
    # roc_curve.fig.set_size_inches((12, 8))
    # roc_curve.savefig('./images/roc_curve.png')

    # plt.plot(fpr, tpr, label="AUC=" + str(auc))
    # plt.xlabel('FPR')
    # plt.ylabel('TPR')
    # plt.legend(loc=4)
    # plt.savefig('roc_curve.png')
