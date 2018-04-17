import numpy
from skimage import io
import copy
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
from tifffile import *
np.set_printoptions(threshold=np.inf)


class ImageData:

    full_left_image = ''
    full_left_image_copy = ''
    full_right_image = ''
    full_right_label =''
    single_left_image = ''
    single_right_image = ''
    num_of_slice = ''
    quantity_of_slices = ''
    single_left_image_copy = ''

    def __init__(self, out_rows, out_cols, data_path="../data/train", label_path="../data/label",
                 test_path="../data/test", npy_path="../data/npydata", img_type=".tif"):

        self.out_rows = out_rows        # rows number
        self.out_cols = out_cols        # columns number
        self.data_path = data_path      # path to train data
        self.label_path = label_path    # path to answers data
        self.img_type = img_type        # image type
        self.test_path = test_path      # path to test data
        self.npy_path = npy_path        # path to numpy files

        # loading default left and right images
        #self.set_left('../data/train-volume.tif')
        #self.set_right('../data/train-volume.tif')

    def create_train_data(self):
        print('-' * 25)
        print('Creating training images')
        print('-' * 25)
        imgs = {}
        for i in range(165):
                imgs[i] = imread(self.data_path + "/" + str(i) + self.img_type)
        print(len(imgs))
        imgdatas = np.ndarray((len(imgs), self.out_rows, self.out_cols, 1), dtype=np.uint8)
        imglabels = np.ndarray((len(imgs), self.out_rows, self.out_cols, 1), dtype=np.uint8)
        for i in range(165):
                img = load_img(self.data_path + "/" + str(i) + self.img_type, grayscale=True)
                label = load_img(self.label_path + "/" + str(i) + self.img_type, grayscale=True)
                img = img_to_array(img)
                label = img_to_array(label)
                imgdatas[i] = img
                imglabels[i] = label
        print('loading done')
        np.save(self.npy_path + '/imgs_train.npy', imgdatas)
        np.save(self.npy_path + '/imgs_mask_train.npy', imglabels)
        print('Saving to .npy files done.')

    def create_train_data_2(self):
        print('-' * 25)
        print('Creating training images')
        print('-' * 25)
        imgs = {}
        for j in range(165):
            for i in range(8):
                imgs[j*8+i] = imread(self.data_path + "/cut_train" + str(i) + "_" + str(j) + self.img_type)
        print(len(imgs))
        imgdatas = np.ndarray((len(imgs), self.out_rows, self.out_cols, 1), dtype=np.uint8)
        imglabels = np.ndarray((len(imgs), self.out_rows, self.out_cols, 1), dtype=np.uint8)
        for j in range(165):
            for i in range(8):
                img = load_img(self.data_path + "/cut_train" + str(i) + "_" + str(j) + self.img_type, grayscale=True)
                label = load_img(self.label_path + "/cut_label" + str(i) + "_" + str(j) + self.img_type, grayscale=True)
                img = img_to_array(img)
                label = img_to_array(label)
                imgdatas[j*8+i] = img
                imglabels[j*8+i] = label
        print('loading done')
        np.save(self.npy_path + '/imgs_train.npy', imgdatas)
        np.save(self.npy_path + '/imgs_mask_train.npy', imglabels)
        print('Saving to .npy files done.')

    def create_train_data_3(self):
        print('-' * 25)
        print('Creating training images')
        print('-' * 25)
        imgs = {}
        for i in range(165*6):
                imgs[i] = imread("../data/cut_e_mirror_train/cut_e_mirror_train_" + str(i) + self.img_type)
        print(len(imgs))
        imgdatas = np.ndarray((len(imgs), self.out_rows, self.out_cols, 1), dtype=np.uint8)
        imglabels = np.ndarray((len(imgs), 388, 388, 1), dtype=np.uint8)
        for j in range(165):
            for i in range(6):
                img = load_img("../data/cut_e_mirror_train/cut_e_mirror_train_" + str(j*6+i) + self.img_type, grayscale=True)
                label = load_img(self.label_path + "/cut_label" + str(i) + "_" + str(j) + self.img_type, grayscale=True)
                img = img_to_array(img)
                label = img_to_array(label)
                imgdatas[j*6+i] = img
                imglabels[j*6+i] = label
        print('loading done')
        np.save(self.npy_path + '/imgs_train.npy', imgdatas)
        np.save(self.npy_path + '/imgs_mask_train.npy', imglabels)
        print('Saving to .npy files done.')

    def create_test_data(self):
        i = 0
        print('-' * 21)
        print('Creating test images')
        print('-' * 21)
        imgs = {}
        for i in range(165):
            imgs[i] = imread(self.test_path + "/" + str(i) + self.img_type)
        print(len(imgs))
        imgdatas = np.ndarray((len(imgs), 768, 1024, 1), dtype=np.uint8)
        for i in range(165):
            img = load_img(self.test_path + "/" + str(i) + self.img_type, grayscale=True)
            img = img_to_array(img)
            imgdatas[i] = img
            i += 1
        print('loading done')
        np.save(self.npy_path + '/imgs_test.npy', imgdatas)
        print('Saving to imgs_test.npy files done.')

    def create_test_data_2(self):
        print('-' * 21)
        print('Creating test images')
        print('-' * 21)
        imgs = {}
        for i in range(165):
                imgs[i] = imread(self.test_path + "/" + str(i) + self.img_type)
        print(len(imgs))
        imgdatas = np.ndarray((len(imgs), self.out_rows, self.out_cols, 1), dtype=np.uint8)
        for i in range(165):
                img = load_img(self.test_path + "/" + str(i) + self.img_type,
                               grayscale=True)
                img = img_to_array(img)
                imgdatas[i] = img
        print('loading done')
        np.save(self.npy_path + '/imgs_test.npy', imgdatas)
        print('Saving to imgs_test.npy files done.')

    def load_train_data(self):
        print('-' * 18)
        print('load train images')
        print('-' * 18)
        imgs_train = np.load(self.npy_path + "/imgs_train.npy")
        imgs_mask_train = np.load(self.npy_path + "/imgs_mask_train.npy")
        imgs_train = imgs_train.astype('float32')
        imgs_mask_train = imgs_mask_train.astype('float32')
        imgs_train /= 255
        imgs_mask_train /= 255
        imgs_mask_train[imgs_mask_train > 0.5] = 1
        imgs_mask_train[imgs_mask_train <= 0.5] = 0
        return imgs_train, imgs_mask_train

    def load_test_data_2(self):
        print('-' * 18)
        print('load test images')
        print('-' * 18)
        imgs_test = np.load(self.npy_path + "/imgs_test.npy")
        imgs_test = imgs_test.astype('float32')
        imgs_test /= 255
        return imgs_test

    def load_test_data(self):
        print('-' * 18)
        print('load test images')
        print('-' * 18)
        imgs_test = np.load(self.npy_path + "/imgs_test_cut.npy")
        imgs_test = imgs_test.astype('float32')
        imgs_test /= 255
        return imgs_test

    def set_left(self, path):
        self.full_left_image = io.imread(path)
        self.full_left_image_copy = copy.deepcopy(self.full_left_image)

        self.num_of_slice = 0
        if numpy.array(self.full_left_image).ndim == 3:
            self.quantity_of_slices = numpy.array(self.full_left_image).shape[0]

        self.single_left_image = self.full_left_image[self.num_of_slice]
        self.single_left_image_copy = self.full_left_image_copy[self.num_of_slice]

        return 0

    def set_right(self, path):
        self.full_right_image = io.imread(path)
        self.single_right_image = self.full_right_image[self.num_of_slice]

        return 0

    def cancel_changes(self):
        self.full_left_image_copy[self.num_of_slice] = copy.deepcopy(
            self.single_left_image)

    def print_single(self):
        print(self.full_left_image[self.num_of_slice])
        return 0

    def print_single_copy(self):
        print(self.full_left_image_copy[self.num_of_slice])
        return 1

    def unet_enable(self):
        self.set_left("../data/test-volume.tif")
        self.set_right('../data/result-label-volume.tif')
        return 0

    def unet_disable(self):
        self.set_left("../data/train-volume.tif")
        self.set_right('../data/result-label-volume.tif')
        return 0


if __name__ == "__main__":
    #x, y = imread("../data/cut_e_mirror_train/cut_e_mirror_train_0.tif").shape
    #mydata = ImageData(256, 256, "../data/cut_train", "../data/cut_label")
    mydata = ImageData(768, 1024, "../data/train", "../data/label")
    mydata.create_train_data()
    mydata.create_test_data_2()
    imgs_train, imgs_mask_train = mydata.load_train_data()
    #print(imgs_train[0])
    #print(imgs_mask_train[0])
    imgs_test = mydata.load_test_data_2()
