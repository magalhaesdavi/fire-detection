import cv2
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(m, classes_names):

    df_cm = pd.DataFrame(m, index=[i for i in classes_names],
                         columns=[i for i in classes_names])
    plt.figure(figsize=(10, 7))

    heatmap = sns.heatmap(df_cm, annot=True, cmap="Blues")
    plt.ylabel("Ground truth")
    plt.xlabel("Prediction")
    plt.title("Normalized confusion matrix")
    fig = heatmap.get_figure()
    # fig.savefig('images/coatnet_4_and_v0_cm.png')
    fig.savefig('test.png')


classes_names = ["fire", "non-fire"]
preds_path = 'output/intersection/coatnet_and_v0/'
gt_path = 'datasets/test/gt/'

if os.path.isdir(preds_path):
    lst_img = [os.path.join(preds_path, file)
               for file in os.listdir(preds_path)]
else:
    if os.path.isfile(preds_path):
        lst_img = [preds_path]
    else:
        raise Exception("Invalid path")

nb_classes = 2
confusion_matrix = torch.zeros(nb_classes, nb_classes)

for im in lst_img:
    print('\t|____Image processing: ', im)

    preds = cv2.imread(im)
    targets = cv2.imread(gt_path + (im[im.rfind("/") + 1:]).replace('.', '_gt.'))

    preds = preds[:, :, 0]
    preds = preds / 255
    preds = preds.reshape(-1)
    targets = targets[:, :, 0]
    targets = targets / 255
    targets = targets.reshape(-1)

    # for t, p in zip(targets, preds):
    #     confusion_matrix[int(t), int(p)] += 1

    y_pred = preds
    y_val = targets

    # 1 = positivo/é fogo
    # 0 = negativo/não é fogo
    FP = len(np.where(y_pred - y_val == 1)[0])  # 10
    FN = len(np.where(y_pred - y_val == -1)[0])  # 01
    TP = len(np.where(y_pred + y_val == 2)[0])  # 11
    TN = len(np.where(y_pred + y_val == 0)[0])  # 00
    confusion_matrix[0, 0] += TP
    confusion_matrix[0, 1] += FN
    confusion_matrix[1, 0] += FP
    confusion_matrix[1, 1] += TN

confusion_matrix = confusion_matrix.numpy()
confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
plot_confusion_matrix(confusion_matrix, classes_names)
