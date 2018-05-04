from skimage import io
from PIL import Image
import numpy
from tifffile import *

numpy.set_printoptions(threshold=numpy.inf)


def split_img(name, dest, num=""):
    img = imread("../data/" + name + num + ".tif")
    for i in range(img.shape[0]):
        imgname = "../data/" + dest + "/" + name + num + "_" + str(i) + ".tif"
        imsave(imgname, img[i])


def merge_result():
    x, y = imread("../data/result" + "/" + str(0) + ".tif").shape
    print(x, y)
    imgarr = numpy.zeros((165, x, y), 'uint8')
    for i in range(165):
        imgarr[i, :, :] = io.imread("../data/result/" + str(i) + ".tif")
    io.imsave("../data/" + "result-label-volume.tif", imgarr)


def merge_train():
    x, y = imread("../data/train" + "/" + str(0) + ".tif").shape
    print(x, y)
    imgarr = numpy.zeros((165, x, y), 'uint8')
    for i in range(165):
        imgarr[i, :, :] = io.imread("../data/train/" + str(i) + ".tif")
    io.imsave("../data/" + "train-volume.tif", imgarr)


def merge_test():
    x, y = imread("../data/test" + "/" + str(0) + ".tif").shape
    print(x, y)
    imgarr = numpy.zeros((165, x, y), 'uint8')
    for i in range(165):
        imgarr[i, :, :] = io.imread("../data/test/" + str(i) + ".tif")
    io.imsave("../data/" + "test-volume.tif", imgarr)


def cut_images2(name):
    x, y = imread("../data/" + name + "/" + str(0) + ".tif").shape

    imgarr_1 = numpy.zeros((165, x, y), 'uint8')

    for i in range(165):
        imgarr_1[i, :, :] = io.imread("../data/" + name + "/" + str(i) + ".tif")

    imgarr_2 = imgarr_1[:, 0:764, 0:1020]
    io.imsave("../data/" + "cut_high_" + name + "0.tif", imgarr_2)


def cut_images(name):
    # 768x1024

    # 768-388 = 380

    # ostatok 8p => 8p vlevo

    # x == 0-388 380-768

    # 1024-388 = 636

    # 636-388 = 248

    # ostatok 140p => 140/2 = 70 v kazhduy storonu

    # y == 0-388 318-706 636-1024

    x, y = imread("../data/" + name + "/" + str(0) + ".tif").shape
    imgarr_1 = numpy.zeros((165, x, y), 'uint8')
    for i in range(165):
        imgarr_1[i, :, :] = io.imread("../data/" + name + "/" + str(i) + ".tif")

    # first x part
    imgarr_2 = imgarr_1[:, 0:388, 0:388]
    io.imsave("../data/" + "cut_" + name + "0.tif", imgarr_2)

    imgarr_2 = imgarr_1[:, 0:388, 318:706]
    io.imsave("../data/" + "cut_" + name + "1.tif", imgarr_2)

    imgarr_2 = imgarr_1[:, 0:388, 636:1024]
    io.imsave("../data/" + "cut_" + name + "2.tif", imgarr_2)

    # second x part

    imgarr_2 = imgarr_1[:, 380:768, 0:388]
    io.imsave("../data/" + "cut_" + name + "3.tif", imgarr_2)

    imgarr_2 = imgarr_1[:, 380:768, 318:706]
    io.imsave("../data/" + "cut_" + name + "4.tif", imgarr_2)

    imgarr_2 = imgarr_1[:, 380:768, 636:1024]
    io.imsave("../data/" + "cut_" + name + "5.tif", imgarr_2)


def cut_for_same_padding(name):

    #768-256 =
    x, y = imread("../data/" + name + "/" + str(0) + ".tif").shape
    imgarr_1 = numpy.zeros((165, x, y), 'uint8')
    for i in range(165):
        imgarr_1[i, :, :] = io.imread("../data/" + name + "/" + str(i) + ".tif")

    # first x part
    imgarr_2 = imgarr_1[:, 0:256, 0:256]
    io.imsave("../data/" + "cut_" + name + "0.tif", imgarr_2)

    imgarr_2 = imgarr_1[:, 0:256, 256:512]
    io.imsave("../data/" + "cut_" + name + "1.tif", imgarr_2)

    imgarr_2 = imgarr_1[:, 0:256, 512:768]
    io.imsave("../data/" + "cut_" + name + "2.tif", imgarr_2)

    imgarr_2 = imgarr_1[:, 0:256, 768:1024]
    io.imsave("../data/" + "cut_" + name + "3.tif", imgarr_2)

    # second x part

    imgarr_2 = imgarr_1[:, 256:512, 0:256]
    io.imsave("../data/" + "cut_" + name + "4.tif", imgarr_2)

    imgarr_2 = imgarr_1[:, 256:512, 256:512]
    io.imsave("../data/" + "cut_" + name + "5.tif", imgarr_2)

    imgarr_2 = imgarr_1[:, 256:512, 512:768]
    io.imsave("../data/" + "cut_" + name + "6.tif", imgarr_2)

    imgarr_2 = imgarr_1[:, 256:512, 768:1024]
    io.imsave("../data/" + "cut_" + name + "7.tif", imgarr_2)


def train_pixels_loss():
    x, y = 572, 572
    x2 = x
    y2 = y
    print(x2, y2)
    for i in range(4):
        x2 = int((x2 - 4) / 2)
        y2 = int((y2 - 4) / 2)
    for i in range(4):
        x2 = int((x2 - 4) * 2)
        y2 = int((y2 - 4) * 2)
    x2 -= 4
    y2 -= 4
    return x - x2, y - y2


def train_mirror_extrapolation():
    x, y = imread("../data/cut_train/cut_train0_0.tif").shape
    print(x, y)
    plus_x, plus_y = train_pixels_loss()
    imgs = numpy.zeros((165 * 6, x, y), 'uint8')
    for i in range(6):
        for j in range(165):
            imgs[i * 165 + j] = io.imread("../data/cut_train/cut_train" + str(i) + "_" + str(j) + ".tif")

    imgarr = numpy.zeros((165 * 6, x + plus_x, y + plus_y), 'uint8')
    imgarr[:, int(plus_x / 2):(int(plus_x / 2) + 388), int(plus_y / 2):(int(plus_y / 2) + 388)] = imgs

    # ******************* main ********************* #
    # top
    temp = imgarr[:, int(plus_x / 2):plus_x, int(plus_y / 2):(int(plus_y / 2) + y)]
    imgarr[:, 0:int(plus_x / 2), int(plus_y / 2):(int(plus_y / 2) + y)] = temp[:, ::-1, :]
    # left
    temp = imgarr[:, int(plus_x / 2):(int(plus_x / 2) + x), int(plus_y / 2):plus_y]
    imgarr[:, int(plus_x / 2):(int(plus_x / 2) + x), 0:int(plus_y / 2)] = temp[:, :, ::-1]
    # right
    temp = imgarr[:, int(plus_x / 2):(int(plus_x / 2) + x), y:(y + int(plus_y / 2))]
    imgarr[:, int(plus_x / 2):(int(plus_x / 2) + x), (int(plus_y / 2) + y):int(y + plus_y)] = temp[:, :, ::-1]
    # bottom
    temp = imgarr[:, x:(x + int(plus_x / 2)), int(plus_y / 2):(int(plus_y / 2) + y)]
    imgarr[:, (int(plus_x / 2) + x):(x + plus_x), int(plus_y / 2):(int(plus_y / 2) + y)] = temp[:, ::-1, :]

    # ******************* lil cubes ********************* #
    # left top
    temp = imgarr[:, int(plus_x / 2):plus_x, int(plus_y / 2):plus_y]
    imgarr[:, 0:int(plus_x / 2), 0:int(plus_y / 2)] = temp[:, ::-1, ::-1]
    # right top
    temp = imgarr[:, int(plus_x / 2):plus_x, y:(y + int(plus_y / 2))]
    imgarr[:, 0:int(plus_x / 2), (y + int(plus_y / 2)):(y + plus_y)] = temp[:, ::-1, ::-1]
    # left bottom
    temp = imgarr[:, x:(x + int(plus_x / 2)), int(plus_y / 2):plus_y]
    imgarr[:, (x + int(plus_x / 2)):(x + plus_x), 0:int(plus_y / 2)] = temp[:, ::-1, ::-1]
    # right bottom
    temp = imgarr[:, x:(x + int(plus_x / 2)), y:(y + int(plus_y / 2))]
    imgarr[:, (x + int(plus_x / 2)):(x + plus_x), (y + int(plus_y / 2)):(y + plus_y)] = temp[:, ::-1, ::-1]

    return imgarr


if __name__ == "__main__":
    # cut_images("train")
    # cut_images("label")
    # for j in range(6):
    #    split_img("cut_train", str(j))
    #    split_img("cut_label", str(j))
    # print(train_mirror_extrapolation()[0][92])
    # print(train_pixels_loss())
    # io.imsave("../data/cut_e_mirror_train.tif", train_mirror_extrapolation())
    # split_img("cut_e_mirror_train")

    """
    cut_images2("train")
    cut_images2("label")
    
    """

    #cut_for_same_padding("train")
    #cut_for_same_padding("label")
    #for j in range(8):
    #    split_img("cut_train", "cut_train", str(j))
    #    split_img("cut_label", "cut_label", str(j))

    merge_train()
    merge_test()
    merge_result()


    # Image.fromarray(train_mirror_extrapolation()[0]).show()
    # train_mirror_extrapolation()
