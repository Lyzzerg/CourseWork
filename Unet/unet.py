from tkinter import Tk

from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, Dropout
from keras.models import *
from keras.preprocessing.image import array_to_img
from skimage import io

from data.ImageData import ImageData


class UNet(object):
    def __init__(self, data):
        self.data = data
        self.data.create_train_data()
        self.train_images, self.train_mask = data.load_train_data()
        self.data.create_test_data()
        self.test_images = data.load_test_data()
        self.img_rows = data.shape[1]
        self.img_cols = data.shape[2]

    def __get_model(self, size):
        inputs = Input((self.img_rows, self.img_cols, 1), name='in')
        s_c1 = size
        s_c2 = s_c1 * 2
        s_c3 = s_c2 * 2
        s_c4 = s_c3 * 2
        s_c5 = s_c4 * 2
        c1 = Conv2D(s_c1, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='conv11')(
            inputs)
        c1 = Conv2D(s_c1, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='conv12')(c1)
        p1 = MaxPooling2D((2, 2), name='pool1')(c1)

        c2 = Conv2D(s_c2, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='conv21')(p1)
        c2 = Conv2D(s_c2, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='conv22')(c2)
        p2 = MaxPooling2D((2, 2), name='pool2')(c2)

        c3 = Conv2D(s_c3, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='conv31')(p2)
        c3 = Conv2D(s_c3, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='conv32')(c3)
        p3 = MaxPooling2D((2, 2), name='pool3')(c3)

        c4 = Conv2D(s_c4, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='conv41')(p3)
        c4 = Conv2D(s_c4, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='conv42')(c4)
        p4 = MaxPooling2D(pool_size=(2, 2), name='pool4')(c4)

        c5 = Conv2D(s_c5, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='conv51')(p4)
        c5 = Conv2D(s_c5, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='conv52')(c5)
        d5 = Dropout(0.5)(c5)

        u6 = Conv2DTranspose(s_c4, (2, 2), strides=(2, 2), padding='same')(d5)
        u6 = concatenate([u6, c4], axis=3, name='concat1')
        c6 = Conv2D(s_c4, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='conv61')(u6)
        c6 = Conv2D(s_c4, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='conv62')(c6)

        u7 = Conv2DTranspose(s_c3, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = concatenate([u7, c3], axis=3, name='concat2')
        c7 = Conv2D(s_c3, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='conv71')(u7)
        c7 = Conv2D(s_c3, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='conv72')(c7)

        u8 = Conv2DTranspose(s_c2, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = concatenate([u8, c2], axis=3, name='concat3')
        c8 = Conv2D(s_c2, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='conv81')(u8)
        c8 = Conv2D(s_c2, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='conv82')(c8)

        u9 = Conv2DTranspose(s_c1, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = concatenate([u9, c1], axis=3, name='concat4')
        c9 = Conv2D(s_c1, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='conv91')(u9)
        c9 = Conv2D(s_c1, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='conv92')(c9)
        c9 = Conv2D(2, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='conv93')(c9)

        outputs = Conv2D(1, (1, 1), activation='sigmoid', name='out')(c9)

        model = Model(inputs=[inputs], outputs=[outputs])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train(self, callback_name, flkernel_count, batch_size, epochs):
        """
        :param callback_name: name for model weights file
        :param flkernel_count: first layer kernels count (every next layer kernels_count is previous*2)
        :param batch_size: size of batch
        :param epochs: epochs count
        :return:
        """
        print("loading data")
        print("loading data done")
        model = self.__get_model(flkernel_count)
        print("got unet")
        model_checkpoint = ModelCheckpoint("../Unet/weights/" + callback_name + '.hdf5', monitor='loss', verbose=1,
                                           save_best_only=True)
        print('Fitting model...')
        model.fit(self.train_images, self.train_mask, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.2,
                  shuffle=True, callbacks=[model_checkpoint])

    def get_model_with_weights(self, precalc_weights_fname, flkernels_count):
        model = self.__get_model(flkernels_count)
        model.load_weights("../Unet/weights/" + precalc_weights_fname + ".hdf5")
        return model

    def predict(self, model):
        imgs_mask_test = model.predict(self.test_images, verbose=1, batch_size=1)
        np.save('../data/npydata/imgs_mask_test.npy', imgs_mask_test)
        self.save_img()

    def get_intermediate_layer_images(self, callback_name, flkernels_count, layer_name):
        model = self.get_model_with_weights(callback_name, flkernels_count)
        intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
        intermediate_output = intermediate_layer_model.predict(self.test_images[0:10], batch_size=1, verbose=1)
        x, y, z, k = intermediate_output.shape
        print(x, y, z, k)
        res = np.zeros((k, y, z, 1))
        for i in range(k):
            res[i, :, :, 0] = intermediate_output[0, ..., i]
        np.save('../data/npydata/imgs_mask_test.npy', res)
        self.save_img()

    def save_img(self):
        print("saving images")
        imgs = np.load('../data/npydata/imgs_mask_test.npy')
        imgs = imgs.astype('float32')
        imgs *= 255
        x, y, z, _ = imgs.shape
        imgarr = np.zeros((x, y, z), 'uint8')
        for i in range(imgs.shape[0]):
            imgarr[i] = array_to_img(imgs[i])
        io.imsave("../data/result-label-volume.tif", imgarr)
        self.data.load_data("../data/result-label-volume.tif", self.data.datatypes[3])


if __name__ == '__main__':
    callback_name = "unet_768x1024_16_10_1"
    flkernels_count = 16
    epochs = 10
    batch_size = 1
    root = Tk()
    data = ImageData()
    unet = UNet(data)
    # unet.train(callback_name, flkernels_count, batch_size, epochs)

    """
    flk = [8]

    for i in flk:
        callback_name = "256x256_k" + str(i) + "_e5_b4"
        train_unet(x_t, y_t, callback_name, i, batch_size, epochs)

    for i in flk:
        callback_name = "256x256_k" + str(i) + "_e10_b4"
        train_unet(x_t, y_t, callback_name, i, 10, epochs)
    """
    #unet.get_intermediate_layer_images(callback_name, flkernels_count, "pool2")
    unet.predict(unet.get_model_with_weights("unet_768x1024_16_10_1", 16))
    """
    flkernels_count =   16
    batch_size = 1
    callback_name = "256x256_k" + str(flkernels_count) + "_e5_b" + str(batch_size)
    layer_name = "out"
    get_intermediate_layer_input(x_p, y_p, callback_name, flkernels_count, layer_name)
    """
