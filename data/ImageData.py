from tkinter import Tk

from PIL import ImageTk, Image
from skimage import io
import numpy as np
np.set_printoptions(threshold=np.inf)


class ImageData:
    datatypes = ("Train Images", "Train Labels", "Test Images", "Test Labels")

    def __init__(self):
        self.current_num = 0
        self.load_data("../data/train-volume.tif", self.datatypes[0])
        self.load_data("../data/label-volume.tif", self.datatypes[1])
        self.load_data("../data/test-volume.tif", self.datatypes[2])
        self.current_data = self.test_images
        self.shape = self.current_data.shape
        self.__image_resize()

    def look_at(self, datatype):
        if datatype in self.datatypes:
            if datatype == self.datatypes[0]:
                self.current_data = self.train_images
            else:
                if datatype == self.datatypes[1]:
                    self.current_data = self.train_labels
                else:
                    if datatype == self.datatypes[2]:
                        self.current_data = self.test_images
                    else:
                        if datatype == self.datatypes[3]:
                            self.current_data = self.test_labels
            self.shape = self.current_data.shape
        self.__image_resize()

    def __image_resize(self):
        self.img = ImageTk.PhotoImage(Image.fromarray
                                      (self.current_data[self.current_num]).resize
                                      ((640, 480), Image.ANTIALIAS))

    def __change_current_num(self, inc: bool):
        if inc:
            if self.current_num < (self.shape[0] - 1):
                self.current_num += 1
        else:
            if self.current_num > 0:
                self.current_num -= 1

        self.__image_resize()

    def next(self):
        self.__change_current_num(True)

    def previous(self):
        self.__change_current_num(False)

    def load_data(self, filename, datatype):
        if datatype in self.datatypes:
            if datatype == self.datatypes[0]:
                self.train_images = io.imread(filename)
                self.create_train_images()
            else:
                if datatype == self.datatypes[1]:
                    self.train_labels = io.imread(filename)
                    self.create_train_labels()
                else:
                    if datatype == self.datatypes[2]:
                        self.test_images = io.imread(filename)
                        self.shape = self.test_images.shape
                        self.create_test_data()
                    else:
                        if datatype == self.datatypes[3]:
                            self.test_labels = io.imread(filename)
            self.look_at(datatype)

    #U-Net Data Processing

    def create_train_images(self):
        imgdatas = self.train_images
        imgdatass = io.imread("../data/test-volume.tif")
        x, y, z = imgdatas.shape
        imgdatas2 = np.ndarray((x+x, y, z, 1), dtype=np.uint8)
        imgdatas2[0:x, :, :, 0] = imgdatas
        imgdatas2[x:x+x, :, :, 0] = imgdatass
        np.save("../data/npydata/imgs_train.npy", imgdatas2)
        return 1

    def create_train_labels(self):
        imglabels = self.train_labels
        imglabelss = io.imread("../data/test-label.tif")
        x, y, z = imglabels.shape
        imglabels2 = np.ndarray((x+x, y, z, 1), dtype=np.uint8)
        imglabels2[0:x, :, :, 0] = imglabels
        imglabels2[x:x+x, :, :, 0] = imglabelss
        np.save("../data/npydata/imgs_mask_train.npy", imglabels2)
        return 1

    def create_train_data(self):
        print('-' * 25)
        print('Creating training images')
        print('-' * 25)
        self.create_train_images()
        self.create_train_labels()
        print('loading done')
        print('Saving to .npy files done.')

    def create_test_data(self):
        i = 0
        print('-' * 21)
        print('Creating test images')
        print('-' * 21)
        imgdatas = self.test_images
        x,y,z = imgdatas.shape
        imgdatas2 = np.ndarray((x, y, z, 1), dtype=np.uint8)
        imgdatas2[:, :, :, 0] = imgdatas
        print('loading done')
        np.save("../data/npydata/imgs_test.npy", imgdatas2)
        print('Saving to imgs_test.npy files done.')

    def load_train_data(self):
        print('-' * 18)
        print('load train images')
        print('-' * 18)
        imgs_train = np.load("../data/npydata/imgs_train.npy")
        imgs_mask_train = np.load("../data/npydata/imgs_mask_train.npy")
        imgs_train = imgs_train.astype('float32')
        imgs_mask_train = imgs_mask_train.astype('float32')
        imgs_train /= 255
        imgs_mask_train /= 255
        imgs_mask_train[imgs_mask_train > 0.5] = 1
        imgs_mask_train[imgs_mask_train <= 0.5] = 0
        return imgs_train, imgs_mask_train

    def load_test_data(self):
        print('-' * 18)
        print('load test images')
        print('-' * 18)
        imgs_test = np.load("../data/npydata" + "/imgs_test.npy")
        imgs_test = imgs_test.astype('float32')
        imgs_test /= 255
        return imgs_test


if __name__ == "__main__":
    root = Tk()
    mydata = ImageData()
    mydata.create_train_data()
    mydata.create_test_data()
    imgs_train, imgs_mask_train = mydata.load_train_data()
    imgs_test = mydata.load_test_data()
