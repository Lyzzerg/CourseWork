from tkinter import Tk

from Unet.unet import *
from data.ImageData import ImageData
from interface.GUI import GUI

_author_ = 'Eugene Baranov'
_project_ = 'Course Work'


root1 = Tk()
image_data = ImageData()
network = UNet(image_data)
main_window = GUI(root1, image_data, network)
main_window.start()
