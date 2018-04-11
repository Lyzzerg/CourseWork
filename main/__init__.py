from tkinter import Tk
from tkinter.filedialog import askopenfilename, asksaveasfilename

import scipy.misc
from PIL import ImageTk, Image

from Unet.unet import UNet
from data.ImageData import ImageData
from interface.GUI import GUI
from filters.Filters import Filter

_author_ = 'Eugene Baranov'
_project_ = 'Course Work'


def window_renew():
    global window1
    global image_data

    image1 = ImageTk.PhotoImage(Image.fromarray
                                (image_data.single_left_image).resize
                                ((640, 480), Image.ANTIALIAS))
    image2 = ImageTk.PhotoImage(Image.fromarray
                                (image_data.single_left_image_copy).resize
                                ((640, 480), Image.ANTIALIAS))

    window1.change_first_label(image1)
    window1.change_second_label(image2)
    window1.change_third_label(image_data.num_of_slice)


def cancel_changes():
    global window1
    global filters

    window1.change_second_label(filters.cancel_changes())


def open_file():
    global image_data
    global window1

    file_opened = 0

    while file_opened == 0:
        try:
            filename = askopenfilename(initialdir='../', title='choose your image',
                                       filetypes=[('All files', '*.*'), ('TIF files', '*.tif')])
            image_data.set_left(filename)

            image1 = ImageTk.PhotoImage(Image.fromarray
                                        (image_data.single_left_image).resize
                                        ((640, 480), Image.ANTIALIAS))
            image2 = ImageTk.PhotoImage(Image.fromarray
                                        (image_data.single_left_image_copy).resize
                                        ((640, 480), Image.ANTIALIAS))

            window1.change_first_label(image1)
            window1.change_second_label(image2)
            window1.change_third_label(image_data.num_of_slice)

            file_opened = 1
        except FileNotFoundError:
            file_opened = 1
        except OSError:
            file_opened = 0


def save_file():
    global image_data

    try:
        save_as = asksaveasfilename(initialdir='../', title='save your image',
                                    filetypes=[('All files', '*.*'), ('TIF file', '*.tif'),
                                               ('Jpeg file', '*.jpg')])

        scipy.misc.imsave(save_as, image_data.single_left_image_copy)
    except ValueError:
        return 1
    return 0


def but_next(event):
    global image_data
    global window1
    global unet_enabled

    if image_data.num_of_slice < image_data.quantity_of_slices - 1:
        image_data.num_of_slice = image_data.num_of_slice + 1
        image_data.single_left_image = image_data.full_left_image[image_data.num_of_slice]
        image_data.single_left_image_copy = image_data.full_left_image_copy[image_data.num_of_slice]

        window_renew()


def but_previous(event):
    global image_data
    global window1

    if image_data.num_of_slice > 0:
        image_data.num_of_slice = image_data.num_of_slice - 1

        image_data.single_left_image = image_data.full_left_image[image_data.num_of_slice]
        image_data.single_left_image_copy = image_data.full_left_image_copy[image_data.num_of_slice]

        window_renew()


def invert():
    global window1
    global filters

    window1.change_second_label(filters.invert_filter())

    return 1


def gradual():
    global window1
    gradual_c = window1.new_window('Enter C in C*r^Y')
    gradual_y = window1.new_window('Enter Y in C*r^Y')
    window1.change_second_label(filters.gradual_filter(gradual_c, gradual_y))
    return 1


def logarithmic():
    global window1
    logarithmic_c = window1.new_window('Enter C in C*log(1+r)')
    window1.change_second_label(filters.logarithmic_filter(logarithmic_c))
    return 1


def unet_enable():
    global unet_enabled
    global filters
    global model
    global image_data

    image_data.unet_enable()
    unet_enabled = True

    #imgs_test = image_data.load_test_data()
    #imgs_mask_test = model.predict(imgs_test, verbose=1, batch_size=1)
    #np.save('../data/npydata/imgs_mask_test.npy', imgs_mask_test)
    #myunet.save_img()
    window_renew()


def analyze_func(event):
    global model
    global image_data
    global filters
    global unet_enabled
    if unet_enabled:
        image_data.single_left_image_copy = image_data.full_left_image_copy[image_data.num_of_slice]
        window_renew()
        window1.change_second_label(filters.illuminate())


def unet_disable():
    global unet_enabled
    image_data.unet_disable()
    unet_enabled = False
    window_renew()


unet_enabled = False

root1 = Tk()

window1 = GUI(root1, open_file, save_file,
              invert, gradual, logarithmic, cancel_changes,
              but_next, but_previous, unet_enable, unet_disable, analyze_func)

image_data = ImageData(768, 1024)

myunet = UNet()
model = myunet.get_unet()
model.load_weights('../Unet/unet.hdf5')

filters = Filter(image_data)

left_image = ImageTk.PhotoImage(Image.fromarray
                          (image_data.single_left_image).resize
                          ((640, 480), Image.ANTIALIAS))
right_image = ImageTk.PhotoImage(Image.fromarray
                          (image_data.single_left_image_copy).resize
                          ((640, 480), Image.ANTIALIAS))

window1.change_first_label(left_image)
window1.change_second_label(right_image)

window1.start()
