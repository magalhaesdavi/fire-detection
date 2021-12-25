import cv2
import numpy as np
from PIL import Image
import os


# CALCULA A IOU MÉDIA DA MASCARA DO COLOR_CLASSIFIER COM O GROUND TRUTH
# img = cv2.imread('images/mask.jpg')
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# output = gray_img
#
# img = cv2.imread('../segmentation_dataset/ground_truth/fire000_gt.png')
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# label = gray_img
# # print(type(gray_img))
# # cv2.imwrite('test.png', gray_img)
#
# SMOOTH = 1e-6
# # print(output.shape)
# # output = output.squeeze(1)
#
# intersection = (output & label).sum()
# # print(intersection)
# union = (output | label).sum()
# # print(union)
#
# iou = (intersection + SMOOTH) / (union + SMOOTH)
#
# inverse_output = 1 - output
# inverse_label = 1 - label
#
# intersection = (output & label).sum()
# # print(intersection)
# union = (output | label).sum()
# # print(union)
#
# iou += (intersection + SMOOTH) / (union + SMOOTH)
# iou /= 2
#
# # thresholded = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10
#
# print(iou)

# TRANSFORMA IMAGEM EM BINARIA
# img = np.array(Image.open('images/mask.jpg'))
# # print(img.shape)
# _, bin_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
# # print(type(bin_img))
#
# # output = Image.fromarray(bin_img)
# # im.show()
# output = bin_img
# # print(output.shape)
#
# img = np.array(Image.open('../segmentation_dataset/ground_truth/fire000_gt.png'))
# _, bin_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
# # binarr = np.where(img>128, 255, 0)
# # Covert numpy array back to image
# # binimg = Image.fromarray(binarr)
#
# label = Image.fromarray(bin_img)
# label.show()
# # label = bin_img
# label = bin_img
# # print(label)
# # print(label.shape)


input_path = 'output/color_classifier/v0/'
input_path2 = 'output/model/coatnet_4/'

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
    img1 = cv2.imread(im)
    # print(img1.shape)
    # cv2.imshow('image', img1)
    # cv2.waitKey(0)
    img2_path = input_path2 + im[im.rfind("/") + 1:]
    # print(img2_path)
    img2 = cv2.imread(img2_path)
    # cv2.imshow('image', img2)
    # cv2.waitKey(0)
    # print(img2.shape)
    dest_and = cv2.bitwise_and(img2, img1, mask=None)
    output_path = 'output/intersection/coatnet_4_and_v0/' + im[im.rfind("/") + 1:]
    cv2.imwrite(output_path, dest_and)
