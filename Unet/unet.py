from PIL import Image
from tifffile import imread
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D, concatenate, UpSampling2D, Cropping2D, Conv2DTranspose, Dropout
from keras.models import *
from keras.optimizers import *
from keras.preprocessing.image import array_to_img

from data.ImageData import ImageData


class UNet(object):
    def __init__(self, img_rows=768, img_cols=1024):
        self.img_rows = img_rows
        self.img_cols = img_cols

    def load_data(self):
        mydata = ImageData(self.img_rows, self.img_cols, "../data/train", "../data/label")
        imgs_train, imgs_mask_train = mydata.load_train_data()
        # imgs_test = mydata.load_test_data()
        return imgs_train, imgs_mask_train  # , imgs_test

    def mean_iou(self, y_true, y_pred):
        prec = []
        for t in np.arange(0.5, 1.0, 0.05):
            y_pred_ = tf.to_int32(y_pred > t)
            score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
            K.get_session().run(tf.local_variables_initializer())
            with tf.control_dependencies([up_opt]):
                score = tf.identity(score)
            prec.append(score)
        return K.mean(K.stack(prec), axis=0)

    def get_unet_same_padding(self, size):

        inputs = Input((self.img_rows, self.img_cols, 1), name='in')
        s_c1 = size
        s_c2 = s_c1 * 2
        s_c3 = s_c2 * 2
        s_c4 = s_c3 * 2
        s_c5 = s_c4 * 2
        c1 = Conv2D(s_c1, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='conv11')(inputs)
        c1 = Conv2D(s_c1, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='conv12')(c1)
        p1 = MaxPooling2D((2, 2), name='pool1')(c1)

        c2 = Conv2D(s_c2, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='conv21')(p1)
        c2 = Conv2D(s_c2, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='conv2')(c2)
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

        outputs = Conv2D(1, (1, 1), activation='sigmoid', name='out')(c9)

        model = Model(inputs=[inputs], outputs=[outputs])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train(self, name, size):
        print("loading data")
        # imgs_train, imgs_mask_train, imgs_test = self.load_data()
        imgs_train, imgs_mask_train = self.load_data()
        print("loading data done")
        model = self.get_unet_same_padding(size)
        print("got unet")

        model_checkpoint = ModelCheckpoint(name+'.hdf5', monitor='loss', verbose=1, save_best_only=True)
        print('Fitting model...')
        model.fit(imgs_train, imgs_mask_train, batch_size=1, epochs=10, verbose=1, validation_split=0.2, shuffle=True,
                  callbacks=[model_checkpoint])

        # print('predict test data')
        # imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
        # np.save('../data/npydata/imgs_mask_test.npy', imgs_mask_test)

    def save_img(self):
        print("saving images")
        imgs = np.load('../data/npydata/imgs_mask_test.npy')

        imgs = imgs.astype('float32')
        #imgs[imgs >= 0.5] = 1
        #imgs[imgs < 0.5] = 0
        imgs *= 255
        for i in range(imgs.shape[0]):
            img = imgs[i]
            img = array_to_img(img)
            img.save("../data/result/" + str(i) + ".tif")


if __name__ == '__main__':
    #x, y = imread("../data/cut_e_mirror_train/cut_e_mirror_train_0.tif").shape
    #print(x, y)
    myunet = UNet(768, 1024)
    #myunet = UNet(256, 256)
    myunet.train('unet_1024_20_10_1_d5_a', 20)
    #myunet = UNet(768, 1024)
    model = myunet.get_unet_same_padding(20)
    model.load_weights('unet_1024_20_10_1_d5_a.hdf5')

    #mydata = ImageData(768, 1024)
    mydata = ImageData(256, 256, "../data/cut_train", "../data/cut_label")
    # imgs_train, imgs_mask_train = mydata.load_train_data()
    imgs_test = mydata.load_test_data_2()
    # imgs_test, _ = mydata.load_test_data_2()

    #img = array_to_img(label_test[0])
    #img.show()

    """
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer('out').output)
    intermediate_output = intermediate_layer_model.predict(imgs_test[0:10], batch_size=1, verbose=1)

    print(intermediate_output.shape)

    x, y, z, _ = intermediate_output.shape

    res = np.zeros((x, y, z, 1))
    res[:, :, :, 0] = intermediate_output[..., 0]
    # res[:, :, :, 0] = 0
    """
    #"""
    imgs_mask_test = model.predict(imgs_test[0:10], verbose=1, batch_size=1)

    # """
    np.save('../data/npydata/imgs_mask_test.npy', imgs_mask_test)
    #np.save('../data/npydata/imgs_mask_test.npy', res)
    myunet.save_img()
