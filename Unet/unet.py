from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, Dropout
from keras.models import *
from keras.preprocessing.image import array_to_img

from data.ImageData import ImageData


class UNet(object):
    def __init__(self, img_rows=768, img_cols=1024):
        self.img_rows = img_rows
        self.img_cols = img_cols

    def load_data(self):
        mydata = ImageData(self.img_rows, self.img_cols, "../data/cut_train", "../data/cut_label")
        imgs_train, imgs_mask_train = mydata.load_train_data()
        return imgs_train, imgs_mask_train

    def get_unet(self, size):
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

    def train(self, name, size, batch, epochs):
        print("loading data")
        imgs_train, imgs_mask_train = self.load_data()
        print("loading data done")
        model = self.get_unet(size)
        print("got unet")
        model_checkpoint = ModelCheckpoint("../Unet/weights/"+name + '.hdf5', monitor='loss', verbose=1, save_best_only=True)
        print('Fitting model...')
        model.fit(imgs_train, imgs_mask_train, batch_size=batch, epochs=epochs, verbose=1, validation_split=0.2,
                  shuffle=True, callbacks=[model_checkpoint])


def save_img():
    print("saving images")
    imgs = np.load('../data/npydata/imgs_mask_test.npy')

    imgs = imgs.astype('float32')
    imgs *= 255
    for i in range(imgs.shape[0]):
        img = imgs[i]
        img = array_to_img(img)
        img.save("../data/result/" + str(i) + ".tif")


def train_unet(x, y, callback_name, flkernels_count, batch_size, epochs):
    """

    :param x: first dimention of image size
    :param y: second dimension of image size
    :param callback_name: name for model weights file
    :param flkernels_count: first layer kernels count (every next layer kernels_count is previous*2)
    :param batch_size: size of batch data
    :param epochs: epochs count
    """
    myunet = UNet(x, y)
    myunet.train(callback_name, flkernels_count, batch_size, epochs)


def load_unet(x, y, callback_name, flkernels_count):
    myunet = UNet(x, y)
    model = myunet.get_unet(flkernels_count)
    model.load_weights("../Unet/weights/" + callback_name + ".hdf5")
    mydata = ImageData(x, y)
    imgs_test = mydata.load_test_data_2()

    return model, imgs_test


def predict_unet(x, y, callback_name, flkernels_count):
    model, imgs_test = load_unet(x, y, callback_name, flkernels_count)
    imgs_mask_test = model.predict(imgs_test, verbose=1, batch_size=1)
    np.save('../data/npydata/imgs_mask_test.npy', imgs_mask_test)
    save_img()


def get_intermediate_layer_input(x, y, callback_name, flkernels_count, layer_name):
    model, imgs_test = load_unet(x, y, callback_name, flkernels_count)
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict(imgs_test[0:10], batch_size=1, verbose=1)
    x, y, z, _ = intermediate_output.shape
    res = np.zeros((x, y, z, 1))
    res[:, :, :, 0] = intermediate_output[..., 0]
    np.save('../data/npydata/imgs_mask_test.npy', res)
    save_img()


if __name__ == '__main__':
    x_t, x_p = 256, 768
    y_t, y_p = 256, 1024

    callback_name = "unet_256_32_5_1_d5_a"
    flkernels_count = 16
    batch_size = 4
    epochs = 5
    """
    flk = [8]

    for i in flk:
        callback_name = "256x256_k" + str(i) + "_e5_b4"
        train_unet(x_t, y_t, callback_name, i, batch_size, epochs)

    for i in flk:
        callback_name = "256x256_k" + str(i) + "_e10_b4"
        train_unet(x_t, y_t, callback_name, i, 10, epochs)
    """
    #predict_unet(x_p, y_p, callback_name, flkernels_count)
    #"""
    flkernels_count =   16
    batch_size = 1
    callback_name = "256x256_k" + str(flkernels_count) + "_e5_b" + str(batch_size)
    layer_name = "out"
    get_intermediate_layer_input(x_p, y_p, callback_name, flkernels_count, layer_name)
    #"""
