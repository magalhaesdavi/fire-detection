from __future__ import print_function
import numpy as np
# from colorthief import ColorThief
import cv2
import os

directory = os.fsencode('../dataset/fire/')

mean_hue_max = 0
mean_hue_min = 0
cont = 0

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    cont += 1
    if filename.endswith(".asm") or filename.endswith(".py"):
        # print(os.path.join(directory, filename))
        continue
    else:
        img = cv2.imread('../dataset/fire/' + filename)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hueMax = hsv_img[:, :, 0].max()
        hueMin = hsv_img[:, :, 0].min()
        mean_hue_max += hueMax
        mean_hue_min += hueMin
        continue

mean_hue_max /= cont
mean_hue_min /= cont
# print(mean_hue_max)
# print(mean_hue_min)

hueMax = mean_hue_max + 10 if mean_hue_max + 10 <= 180 else 180
hueMin = mean_hue_min - 10 if mean_hue_min - 10 >= 0 else 0
# print(hueMax)
# print(hueMin)

# img = cv2.imread('../dataset/fire/206488_218.jpg')

# test = cv2.imread('../segmentation_dataset/data/fire008.png')
# hsv_test = cv2.cvtColor(test, cv2.COLOR_BGR2HSV)

# hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# # print(type(hsv_img))
#
# hueMax = hsv_img[:, :, 0].max()
# hueMin = hsv_img[:, :, 0].min()
# # print(hsv_img[:, :, 0].shape)
# # print(hueMin)
# # print(hueMax)
# # x = hsv_img[:, :, 0]
# # idx = np.unravel_index(np.argmax(x), x.shape)
# # print(idx)
# # print(hsv_img[48, 35, 2])
# # coordinates = 48, 35
# # 150, 1, 255 = #FCFFFE = 252, 255, 254
# # huemin = 0
# # huemax = 150
# # [hueMin, 70, 190]
# # [hueMax+10, 255, 255]

lowerBound = np.array([hueMin, 0, 190], np.uint8)
upperBound = np.array([hueMax, 255, 255], np.uint8)

# mask = cv2.inRange(hsv_test, lowerBound, upperBound)
# cv2.imwrite("../mask.png", mask)

# output_img = cv2.bitwise_and(test, test, mask=mask)
# cv2.imwrite("../output.png", output_img)

input_path = 'D:/Documentos/fire_detection/BoWFireDataset/BoWFireDataset/dataset/img'

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
    img = cv2.imread(im)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_img, lowerBound, upperBound)
    output_path = '../results/mask/color classifier/' + im[im.rfind("\\") + 1:]
    # print(im)
    # print(im[im.rfind("\\") + 1:])
    cv2.imwrite(output_path, mask)


# end
# plt.imshow(mask, cmap='gray')   # this colormap will display in black / white
# plt.show()

# Image filling
# seg_im = Image.new('RGB', (50, 50), color='black')
# d = ImageDraw.Draw(im)
# k = []
# for x in range(0, 50):
#     for y in range(0, 50):
#         pixel = im.getpixel((x, y))
#         pix_rgb = tuple(pixel)
#         if pix_rgb in t:
#             k += [(x, y)]
# d.point(k, fill=(255, 255, 255))
# save = '../segtest.jpg'
# seg_im.save(save)

# color_thief palette
# color_thief = ColorThief('../dataset/train/fire/fire69.jpg')
# # get the dominant color
# dominant_color = color_thief.get_color(quality=1)
# palette = color_thief.get_palette(color_count=6)
# print(palette)

# most frequent colors (snippet from:
# https://stackoverflow.com/questions/3241929/python-find-dominant-most-common-color-in-an-image)
# NUM_CLUSTERS = 1000
#
# print('reading image')
# im = Image.open('../dataset/train/fire/206488_218.jpg')
#
# im = im.convert('RGB')
# # print(im.getpixel((25, 25)))  # Get the RGBA Value of the a pixel of an image
# # im = im.resize((150, 150))      # optional, to reduce time
# ar = np.asarray(im)
# shape = ar.shape
# ar = ar.reshape(scipy.product(shape[:2]), shape[2]).astype(float)
#
# print('finding clusters')
# codes, dist = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS)
# # print('cluster centres:\n', codes)
# # print(codes.astype(int))
# vecs, dist = scipy.cluster.vq.vq(ar, codes)         # assign codes
# counts, bins = scipy.histogram(vecs, len(codes))    # count occurrences
# codes = codes.astype(int)
# t = tuple(map(tuple, codes))
# print(len(t))
#
# index_max = scipy.argmax(counts)                    # find most frequent
# peak = codes[index_max]
# colour = binascii.hexlify(bytearray(int(c) for c in peak)).decode('ascii')
# print('most frequent is %s (#%s)' % (peak, colour))
#
#
# # bonus: save image using only the N most common colours
# c = ar.copy()
# for i, code in enumerate(codes):
#     c[scipy.r_[scipy.where(vecs == i)], :] = code
# imageio.imwrite('../clusters.png', c.reshape(*shape).astype(np.uint8))
# print('saved clustered image')


# Eduardo's code
# import numpy as np
# import os
# from PIL import Image, ImageDraw
# import cv2
# from matplotlib import pyplot
# from mpl_toolkits.mplot3d import Axes3D
#
# def distance(x, y):
#     k = 0
#     menor = True
#     for z in range(0, 3):
#         if x[z] > y[z]:
#             k = x[z] - y[z]
#         else:
#             k = y[z] - x[z]
#         if k > 5:
#             menor = False
#             break
#     return menor
#
#
# stepSize = 1
# setTreino = set()
# setNovo = set()
# fire = 'C:/FireDetection/train/fire'
# files = os.listdir(fire)
# for b in range(0, 80):
#     path = fire + "/" + 'fire' + '{0:0=2d}'.format(b + 1) + '.jpg'
#     print(path)
#     image = cv2.imread(path)
#     for x in range(0, image.shape[1], stepSize):
#         for y in range(0, image.shape[0], stepSize):
#             px = image[y, x]
#             py = tuple(px)
#             setTreino.add(py)
# maior = []
# menor = []
# list_of_lists = [list(elem) for elem in setTreino]
# setNovo.union(setTreino)
# for x in range(len(list_of_lists)):
#     for y in range(x + 1, len(list_of_lists)):
#         if distance(list_of_lists[x], list_of_lists[y]):
#             for z in range(0, 3):
#                 if list_of_lists[x][z] > list_of_lists[y][z]:
#                     maior.append(list_of_lists[x][z])
#                     menor.append(list_of_lists[y][z])
#                 else:
#                     maior.append(list_of_lists[y][z])
#                     menor.append(list_of_lists[x][z])
#             for k in range(menor[0], maior[0] + 1):
#                 for l in range(menor[1], maior[1] + 1):
#                     for m in range(menor[2], maior[2] + 1):
#                         px = [k, l, m]
#                         py = tuple(px)
#                         setNovo.add(py)
#             maior.clear()
#             menor.clear()
# print(len(setNovo), len(setTreino), len(setNovo) - len(setTreino))
#
# k = []
# cont = 0
# fire = 'C:/FireDetection/teste/fire'
# files = os.listdir(fire)
# for fil in files:
#     path = fire + "/" + fil
#     image = cv2.imread(path)
#     im = Image.new('RGB', (image.shape[1], image.shape[0]), color='black')
#     d = ImageDraw.Draw(im)
#     for x in range(0, image.shape[1], stepSize):
#         for y in range(0, image.shape[0], stepSize):
#             px = image[y, x]
#             py = tuple(px)
#             if py in setNovo:
#                 k += [(x, y)]
#     d.point(k, fill=(255, 255, 255))
#     save = 'resultadoFireCor' + '_{0:0=3d}'.format(cont) + '.png'
#     im.save(save)
#     cont += 1
#     k.clear()
#
# cont = 0
# fire = 'C:/FireDetection/teste/normal'
# files = os.listdir(fire)
# for fil in files:
#     path = fire + "/" + fil
#     image = cv2.imread(path)
#     im = Image.new('RGB', (image.shape[1], image.shape[0]), color='black')
#     d = ImageDraw.Draw(im)
#     for x in range(0, image.shape[1], stepSize):
#         for y in range(0, image.shape[0], stepSize):
#             px = image[y, x]
#             py = tuple(px)
#             if py in setNovo:
#                 k += [(x, y)]
#     d.point(k, fill=(255, 255, 255))
#     save = 'resultadoNormalCor' + '_{0:0=3d}'.format(cont) + '.png'
#     im.save(save)
#     cont += 1
#     k.clear()
#
# '''
# list_of_lists = [list(elem) for elem in setf]
#
# fig = pyplot.figure()
# ax = Axes3D(fig)
# b = []
# g = []
# r = []
# for elem in list_of_lists:
#         k , l , m = elem
#         b.append(k)
#         g.append(l)
#         r.append(m)
# array = []
# npArray = np.array([np.array(xi) for xi in list_of_lists])
# for elem in npArray:
#         array.append(elem[::-1])
# nArray = np.array([np.array(xi) for xi in array])
# ax.scatter(b, g, r, facecolors = nArray.reshape(-1,3)/255.)
#
# ax.set_xlabel('Blue')
# ax.set_ylabel('Green')
# ax.set_zlabel('Red')
#
# pyplot.show()
# '''
