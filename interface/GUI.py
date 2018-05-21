from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter.messagebox import askyesno

from Unet.unet import UNet
from data.ImageData import ImageData
import os


class GUI:

    def __init__(self, root: Tk, data: ImageData, network: UNet):
        self.network = network
        self.data = data
        self.root = root
        self.root.resizable(width=False, height=False)
        self.root.geometry('{}x{}'.format(1400, 520))
        self.add_labels()
        self.add_buttons()
        self.add_first_menu()
        self.add_second_menu()

    def start(self):
        self.root.mainloop()

    def add_buttons(self):
        self.button_next = Button(text='Next Slice')
        self.button_previous = Button(text='Previous Slice')
        self.button_analyze = Button(text='Analyze')
        self.button_next.bind('<Button-1>', self.next_button_func)
        self.button_previous.bind('<Button-1>', self.previous_button_func)
        self.button_analyze.bind('<Button-1>', self.analyze_button_func)
        self.button_previous.pack(side=BOTTOM)
        self.button_next.pack(side=BOTTOM)
        self.button_analyze.pack(side=TOP)
        return 1

    def add_labels(self):
        self.label = Label(self.root)
        self.label2 = Label(self.root)
        self.label3 = Label(self.root, text='Slice №\n')
        self.label.pack(side=LEFT)
        self.label2.pack(side=RIGHT)
        self.label3.pack(side=TOP)
        return 1

    def add_first_menu(self):
        self.main_menu = Menu(self.root)
        self.root.config(menu=self.main_menu)
        self.first_menu = Menu(self.main_menu)
        self.main_menu.add_cascade(label='Load Data', menu=self.first_menu)
        self.first_menu.add_command(label='Train Images', command=self.open_train_images)
        self.first_menu.add_command(label='Train Labels', command=self.open_train_labels)
        self.first_menu.add_command(label='Test Images', command=self.open_test_images)
        return 1

    def add_second_menu(self):
        self.second_menu = Menu(self.main_menu)
        self.main_menu.add_cascade(label='Show Data', menu=self.second_menu)
        self.second_menu.add_command(label='Train Images', command=self.show_train_images)
        self.second_menu.add_command(label='Train Labels', command=self.show_train_labels)
        self.second_menu.add_command(label='Test Images', command=self.show_test_images)
        self.second_menu.add_command(label='Test Labels', command=self.show_test_labels)
        return 1

    def __labels_renew(self):
        self.label.configure(image=self.data.img)
        self.label2.configure(image=self.data.img)
        self.label3.configure(text='Slice №\n' + str(self.data.current_num + 1))

    def next_button_func(self, event):
        self.data.next()
        self.__labels_renew()
        return 1

    def previous_button_func(self, event):
        self.data.previous()
        self.__labels_renew()
        return 1

    def analyze_button_func(self, event):
        weights_name, flkernel_size, layername = self.nn_params_window()
        self.network.get_intermediate_layer_images(weights_name, flkernel_size, layername)
        # self.network.get_intermediate_layer_images("unet_768x1024_16_10_1", 16, "pool2")

    def __open(self, datatype):
        file_opened = 0

        while file_opened == 0:
            try:
                filename = askopenfilename(initialdir='../', title="Select " + datatype,
                                           filetypes=[('TIF files', '*.tif')])
                file_opened = 1

                self.data.load_data(filename, datatype)
            except FileNotFoundError:
                file_opened = 1
            except OSError:
                file_opened = 0

        return file_opened

    def __show(self, datatype):
        self.data.look_at(datatype)
        self.__labels_renew()

    def show_train_images(self):
        self.__show(self.data.datatypes[0])

    def show_train_labels(self):
        self.__show(self.data.datatypes[1])

    def show_test_images(self):
        self.__show(self.data.datatypes[2])

    def show_test_labels(self):
        self.__show(self.data.datatypes[3])

    def open_train_images(self):
        self.__open(self.data.datatypes[0])

    def open_train_labels(self):
        self.__open(self.data.datatypes[1])

    def open_test_images(self):
        self.__open(self.data.datatypes[2])

    def nn_params_window(self):

        flkernel_size = StringVar()
        layername = StringVar()

        modal_window = Toplevel(self.root)

        modal_frame = Frame(modal_window)

        dir = '../Unet/weights'
        files = os.listdir(dir)
        print(files)

        weights_name_list = Listbox(modal_frame)
        for i in files:
            weights_name_list.insert(0, i[:-5])
        flkernel_size_textbox = Entry(modal_frame, textvariable=flkernel_size)
        layername_textbox = Entry(modal_frame, textvariable=layername)

        button_ok = Button(modal_frame, text='OK')
        result1, result2, result3 = '', '', ''

        def close_modal(event):
            nonlocal result1, result2, result3
            result1 = str(weights_name_list.get(ACTIVE))
            print(result1)
            try:
                result2 = int(flkernel_size_textbox.get())
            except ValueError:
                flkernel_size.set('Wrong input!')
            try:
                result3 = str(layername_textbox.get())
                modal_window.destroy()
            except ValueError:
                layername.set('Wrong input!')

        button_ok.bind('<Button-1>', close_modal)

        modal_frame.pack()

        modal_window.wm_title("Neural network params")
        modal_window.focus_force()
        modal_window.geometry('300x270')

        weights_name_list.pack()
        flkernel_size_textbox.pack()
        layername_textbox.pack()
        button_ok.pack()

        modal_window.wait_window()

        return result1, result2, result3

    def close_win(self):
        if askyesno('Save program', 'Do u want to save files?'):
            self.save_func()
            self.root.destroy()
        else:
            self.root.destroy()
