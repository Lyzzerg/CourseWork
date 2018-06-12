from tkinter import Tk

from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.layers import Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, Dropout, Reshape
from keras.models import *
from keras.preprocessing.image import array_to_img
from skimage import io

from data.ImageData import ImageData

smooth = 1.


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return smooth - dice_coef(y_true, y_pred)


class UNet(object):
    types = {1, 2, 3, 4}

    def __init__(self, data):
        self.data = data
        self.train_images, self.train_mask = data.load_train_data()
        self.test_images = data.load_test_data()
        self.img_rows = data.shape[1]
        self.img_cols = data.shape[2]

    def __get_model(self, type, size):
        if type in self.types:
            if type == 1:
                return self.__model1(size)
            if type == 2:
                return self.__model2(size)
            if type == 3:
                return self.__model3(size)
            if type == 4:
                return self.__model4(size)
        else:  # default
            return self.__model4(size)

    def __model1(self, size):
        inputs = Input((self.img_rows, self.img_cols, 1), name='in')
        s_c1 = size
        s_c2 = s_c1 * 2

        c1 = Conv2D(s_c1, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='conv11')(
            inputs)
        c1 = Conv2D(s_c1, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='conv12')(c1)
        p1 = MaxPooling2D(pool_size=(2, 2), name='pool1')(c1)

        c2 = Conv2D(s_c2, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='conv21')(p1)
        c2 = Conv2D(s_c2, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='conv22')(c2)
        d2 = Dropout(0.5)(c2)

        u3 = Conv2DTranspose(s_c1, (2, 2), strides=(2, 2), padding='same')(d2)
        u3 = concatenate([u3, c1], axis=3, name='concat1')
        c3 = Conv2D(s_c1, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='conv31')(u3)
        c3 = Conv2D(s_c1, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='conv32')(c3)
        c3 = Conv2D(2, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='conv33')(c3)

        outputs = Conv2D(1, (1, 1), activation='sigmoid', name='out')(c3)

        model = Model(inputs=[inputs], outputs=[outputs])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coef, 'accuracy'])
        return model

    def __model2(self, size):
        inputs = Input((self.img_rows, self.img_cols, 1), name='in')
        s_c1 = size
        s_c2 = s_c1 * 2
        s_c3 = s_c2 * 2

        c1 = Conv2D(s_c1, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='conv11')(
            inputs)
        c1 = Conv2D(s_c1, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='conv12')(c1)
        p1 = MaxPooling2D((2, 2), name='pool1')(c1)

        c2 = Conv2D(s_c2, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='conv21')(p1)
        c2 = Conv2D(s_c2, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='conv22')(c2)
        p2 = MaxPooling2D(pool_size=(2, 2), name='pool2')(c2)

        c3 = Conv2D(s_c3, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='conv31')(p2)
        c3 = Conv2D(s_c3, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='conv32')(c3)
        d3 = Dropout(0.5)(c3)

        u4 = Conv2DTranspose(s_c2, (2, 2), strides=(2, 2), padding='same')(d3)
        u4 = concatenate([u4, c2], axis=3, name='concat1')
        c4 = Conv2D(s_c2, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='conv41')(u4)
        c4 = Conv2D(s_c2, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='conv42')(c4)

        u5 = Conv2DTranspose(s_c1, (2, 2), strides=(2, 2), padding='same')(c4)
        u5 = concatenate([u5, c1], axis=3, name='concat2')
        c5 = Conv2D(s_c1, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='conv51')(u5)
        c5 = Conv2D(s_c1, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='conv52')(c5)
        c5 = Conv2D(2, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='conv53')(c5)

        outputs = Conv2D(1, (1, 1), activation='sigmoid', name='out')(c5)

        model = Model(inputs=[inputs], outputs=[outputs])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coef, 'accuracy'])
        return model

    def __model3(self, size):
        inputs = Input((self.img_rows, self.img_cols, 1), name='in')
        s_c1 = size
        s_c2 = s_c1 * 2
        s_c3 = s_c2 * 2
        s_c4 = s_c3 * 2

        c1 = Conv2D(s_c1, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='conv11')(
            inputs)
        c1 = Conv2D(s_c1, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='conv12')(c1)
        p1 = MaxPooling2D((2, 2), name='pool1')(c1)

        c2 = Conv2D(s_c2, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='conv21')(p1)
        c2 = Conv2D(s_c2, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='conv22')(c2)
        p2 = MaxPooling2D((2, 2), name='pool2')(c2)

        c3 = Conv2D(s_c3, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='conv31')(p2)
        c3 = Conv2D(s_c3, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='conv32')(c3)
        p3 = MaxPooling2D(pool_size=(2, 2), name='pool3')(c3)

        c4 = Conv2D(s_c4, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='conv41')(p3)
        c4 = Conv2D(s_c4, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='conv42')(c4)
        d4 = Dropout(0.5)(c4)

        u5 = Conv2DTranspose(s_c3, (2, 2), strides=(2, 2), padding='same')(d4)
        u5 = concatenate([u5, c3], axis=3, name='concat1')
        c5 = Conv2D(s_c3, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='conv51')(u5)
        c5 = Conv2D(s_c3, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='conv52')(c5)

        u6 = Conv2DTranspose(s_c2, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = concatenate([u6, c2], axis=3, name='concat2')
        c6 = Conv2D(s_c2, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='conv61')(u6)
        c6 = Conv2D(s_c2, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='conv62')(c6)

        u7 = Conv2DTranspose(s_c1, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = concatenate([u7, c1], axis=3, name='concat3')
        c7 = Conv2D(s_c1, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='conv71')(u7)
        c7 = Conv2D(s_c1, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='conv72')(c7)
        c7 = Conv2D(2, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='conv73')(c7)

        outputs = Conv2D(1, (1, 1), activation='sigmoid', name='out')(c7)

        model = Model(inputs=[inputs], outputs=[outputs])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coef, 'accuracy'])
        return model

    def __model4(self, size):
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
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coef, 'accuracy'])
        return model

    def train(self, callback_name, type, flkernel_count, batch_size, epochs):
        """
        :param callback_name: name for model weights file
        :param flkernel_count: first layer kernels count (every next layer kernels_count is previous*2)
        :param batch_size: size of batch
        :param epochs: epochs count
        :return:
        """
        print("loading data")
        print("loading data done")
        model = self.__get_model(type, flkernel_count)
        print("got unet")
        model_checkpoint = ModelCheckpoint("../Unet/models/" + callback_name + '.hdf5', monitor='val_loss', verbose=1,
                                           save_best_only=True)

        tensorboard_checkpoint = TensorBoard(log_dir='../Unet/TensorBoard', histogram_freq=0, write_graph=False,
                                             batch_size=2, write_images=False, write_grads=False)

        es_callback = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0)
        print('Fitting model...')
        model.fit(self.train_images, self.train_mask, batch_size=batch_size, epochs=epochs, verbose=1,
                  validation_split=0.5, shuffle=True, callbacks=[es_callback, model_checkpoint, tensorboard_checkpoint])

    def get_trained_model(self, trained_model_name):
        model = load_model("../Unet/models/" + trained_model_name + ".hdf5")
        return model

    def get_model_weights(self, type, size, trained_model_name):
        model = self.__get_model(type, size)
        model.load_weights("../Unet/models/" + trained_model_name + ".hdf5")
        return model

    def predict(self, model, nn):
        imgs_mask_test = model.predict(self.test_images, verbose=1, batch_size=1)
        np.save('../data/npydata/imgs_mask_test.npy', imgs_mask_test)
        self.save_img(nn)

    def get_intermediate_layer_images(self, callback_name, layer_name):
        model = self.get_trained_model(callback_name)
        intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
        intermediate_output = intermediate_layer_model.predict(self.test_images[0:10], batch_size=1, verbose=1)
        x, y, z, k = intermediate_output.shape
        print(x, y, z, k)
        res = np.zeros((k, y, z, 1))
        for i in range(k):
            res[i, :, :, 0] = intermediate_output[0, ..., i]
        np.save('../data/npydata/imgs_mask_test.npy', res)
        self.save_img()

    def save_img(self,nn):
        print("saving images")
        imgs = np.load('../data/npydata/imgs_mask_test.npy')
        imgs = imgs.astype('float32')
        imgs *= 255
        x, y, z, _ = imgs.shape
        imgarr = np.zeros((x, y, z), 'uint8')
        for i in range(imgs.shape[0]):
            imgarr[i] = array_to_img(imgs[i])
        io.imsave("../data/"+nn+".tif", imgarr)
        #self.data.load_data("../data/result-label-volume.tif", self.data.datatypes[3])


if __name__ == '__main__':
    model_type = 4
    flkernels_count = 8
    epochs = 1000
    batch_size = 2

    root = Tk()
    data = ImageData()
    unet = UNet(data)

    callback_name = "unet_centropy_" + "MT" + str(model_type) + "_" + "K" + str(flkernels_count) + "_" + "E" + \
                    str(epochs) + "_" + "B" + str(batch_size)
    unet.train(callback_name, model_type, flkernels_count, batch_size, epochs)

    # unet.get_intermediate_layer_images(callback_name, flkernels_count, "pool2")
    unet.predict(unet.get_model_weights(model_type, flkernels_count, callback_name), callback_name)