import numpy as np
import cv2
import os

def remove_noise(gray, num):
    y, x = gray.shape
    nearest_neigbours = [[
        np.argmax(
            np.bincount(
                gray[max(i - num, 0):min(i + num, y), max(j - num, 0):min(j + num, x)].ravel()))
        for j in range(x)] for i in range(y)]
    result = np.array(nearest_neigbours, dtype=np.uint8)
    return result

def main():
    input_path = 'output/intersection/coatnet472_preclass0.7_and_v0/'

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
        output_path = 'output/refined/coatnet472_preclass0.7_and_v0/' + im[im.rfind("/") + 1:]

        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # result = remove_noise(gray, 10)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # kernel_sizes = [(3, 3), (5, 5), (7, 7)]
        # for kernel_size in kernel_sizes:
        #     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        #     closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        #     cv2.imwrite(output_path, closing)

        kernel_size = (2, 2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        cv2.imwrite(output_path, closing)

        # cv2.imwrite(output_path, result)

if __name__ == "__main__":
    main()