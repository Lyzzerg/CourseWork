import numpy
from PIL import ImageTk, Image

from data import ImageData


class Filter:
    image_data: ImageData
    img_tk: ImageTk.PhotoImage

    def __init__(self, data_set: ImageData):
        self.image_data = data_set

    def __clamp(self, variable: float, range_a: int, range_b: int):
        if variable >= range_b:
            return range_b
        if variable <= range_a:
            return range_a
        return int(variable)

    def invert_filter(self):
        for element in numpy.nditer(
                self.image_data.full_left_image_copy[self.image_data.num_of_slice], op_flags=['readwrite']):
            element[...] = self.__clamp(255 - element, 0, 255)
        img_tk = ImageTk.PhotoImage(Image.fromarray
                                    (self.image_data.single_left_image_copy).resize
                                    ((640, 480), Image.ANTIALIAS))
        return img_tk

    def gradual_filter(self, gradual_c, gradual_y):
        if gradual_c >= 0:
            for element in numpy.nditer(
                    self.image_data.full_left_image_copy[self.image_data.num_of_slice], op_flags=['readwrite']):
                element[...] = self.__clamp((int(element) ** gradual_y) * gradual_c, 0, 255)
        img_tk = ImageTk.PhotoImage(Image.fromarray
                                    (self.image_data.single_left_image_copy).resize
                                    ((640, 480), Image.ANTIALIAS))
        return img_tk

    def logarithmic_filter(self, logarithmic_c):
        if logarithmic_c >= 0:
            for element in numpy.nditer(
                    self.image_data.full_left_image_copy[self.image_data.num_of_slice], op_flags=['readwrite']):
                element[...] = self.__clamp(numpy.math.log1p(1 + element) * logarithmic_c, 0, 255)
        img_tk = ImageTk.PhotoImage(Image.fromarray
                                    (self.image_data.single_left_image_copy).resize
                                    ((640, 480), Image.ANTIALIAS))
        return img_tk

    def illuminate(self):
        for i in range(0, self.image_data.full_left_image_copy.shape[1]):
            for j in range(0, self.image_data.full_left_image_copy.shape[2]):
                self.image_data.full_left_image_copy[self.image_data.num_of_slice][i][j] = self.__clamp(
                    int(self.image_data.full_left_image[self.image_data.num_of_slice][i][j]) +
                    int(self.image_data.full_right_image[self.image_data.num_of_slice][i][j]), 0, 255)

        img_tk = ImageTk.PhotoImage(Image.fromarray
                                    (self.image_data.single_left_image_copy).resize
                                    ((640, 480), Image.ANTIALIAS))

        return img_tk

    def cancel_changes(self):
        self.image_data.cancel_changes()
        image = ImageTk.PhotoImage(Image.fromarray
                                   (self.image_data.single_left_image_copy).resize
                                   ((640, 480), Image.ANTIALIAS))
        return image
