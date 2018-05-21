from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter.messagebox import askyesno

from Unet.unet import UNet
from data.ImageData import ImageData


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
        self.network.get_intermediate_layer_images("unet_768x1024_16_10_1", 16, "pool2")

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







    def new_window(self, value):
        result = -1

        v = StringVar()

        modal_window = Toplevel(self.root)

        modal_frame = Frame(modal_window)

        input_window = Entry(modal_frame, textvariable=v)

        button_ok = Button(modal_frame, text='OK')

        def close_modal(event):
            nonlocal result
            try:
                result = float(input_window.get())
                modal_window.destroy()
            except ValueError:
                v.set('Wrong input!')

        button_ok.bind('<Button-1>', close_modal)

        modal_frame.pack()

        modal_window.wm_title(value)
        x_ = (modal_window.winfo_screenwidth() - modal_window.winfo_reqwidth()) / 2
        y_ = (modal_window.winfo_screenheight() - modal_window.winfo_reqheight()) / 2
        modal_window.wm_geometry('+%d+%d' % (x_, y_))
        modal_window.focus_force()
        modal_window.geometry('300x50')

        input_window.pack()
        button_ok.pack()

        modal_window.wait_window()

        return result

    def close_win(self):
        if askyesno('Save program', 'Do u want to save files?'):
            self.save_func()
            self.root.destroy()
        else:
            self.root.destroy()
