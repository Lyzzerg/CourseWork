from skimage import io

import numpy
from tifffile import *

dirtype = ("../data/train", "../data/label", "../data/test")


def split_img(name, num):
    img = imread("../data/" + name + num + ".tif")
    for i in range(img.shape[0]):
        imgname = "../data/" + name + "/" + name + num + "_" + str(i) + ".tif"
        imsave(imgname, img[i])


def merge_result():
    x, y = imread("../data/result" + "/" + str(0) + ".tif").shape
    print(x, y)
    imgarr = numpy.zeros((165, x, y), 'uint8')
    for i in range(165):
        imgarr[i, :, :] = io.imread("../data/result/" + str(i) + ".tif")
    print(imgarr)
    io.imsave("../data/" + "result-label-volume.tif", imgarr)


def cut_images(name):
    x, y = imread("../data/" + name + "/" + str(0) + ".tif").shape
    print(x, y)
    imgarr_1 = numpy.zeros((165, x, y), 'uint8')
    for i in range(165):
        imgarr_1[i, :, :] = io.imread("../data/" + name + "/" + str(i) + ".tif")

    x_2 = (int)(x / 2)
    y_2 = (int)(y / 2)
    imgarr_2 = imgarr_1[:, 0:x_2, 0:y_2]
    io.imsave("../data/" + "cut_" + name + "0.tif", imgarr_2)
    imgarr_2 = imgarr_1[:, 0:x_2, y_2:y]
    io.imsave("../data/" + "cut_" + name + "1.tif", imgarr_2)
    imgarr_2 = imgarr_1[:, x_2:x, 0:y_2]
    io.imsave("../data/" + "cut_" + name + "2.tif", imgarr_2)
    imgarr_2 = imgarr_1[:, x_2:x, y_2:y]
    io.imsave("../data/" + "cut_" + name + "3.tif", imgarr_2)


if __name__ == "__main__":
    #cut_images("train")
    #cut_images("label")
    for j in range(4):
        split_img("cut_train", str(j))
        split_img("cut_label", str(j))
