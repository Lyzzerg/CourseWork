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
        self.add_third_menu()

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
        self.second_menu.add_command(label='Train Images1', command=self.show_train_images)
        self.second_menu.add_command(label='Train Labels', command=self.show_train_labels)
        self.second_menu.add_command(label='Test Images', command=self.show_test_images)
        self.second_menu.add_command(label='Test Labels', command=self.show_test_labels)
        return 1

    def add_third_menu(self):
        self.third_menu = Menu(self.main_menu)
        self.main_menu.add_cascade(label='Neural Network', menu=self.third_menu)
        self.third_menu.add_command(label='Train', command=self.train_network)
        self.third_menu.add_command(label='Predict', command=self.predict_test)
        self.third_menu.add_command(label='Intermediate Layer', command=self.predict_intermediate)
        return 1

    def __train_window(self):
        callback_name = StringVar()
        flkernel_count = StringVar()
        batch_size = StringVar()
        epochs = StringVar()

        modal_window = Toplevel(self.root)

        modal_frame = Frame(modal_window)
        modal_window.resizable(width=False, height=False)

        label1 = Label(modal_frame, text='Callback name')
        label2 = Label(modal_frame, text='FL kernel count')
        label3 = Label(modal_frame, text='Batch size')
        label4 = Label(modal_frame, text='Epochs')
        label5 = Label(modal_frame, text='')

        callback_name_textbox = Entry(modal_frame, textvariable=callback_name)
        flkernel_count_textbox = Entry(modal_frame, textvariable=flkernel_count)
        batch_size_textbox = Entry(modal_frame, textvariable=batch_size)
        epochs_textbox = Entry(modal_frame, textvariable=epochs)

        label1.grid(row=0, column=0)
        label2.grid(row=1, column=0)
        label3.grid(row=2, column=0)
        label4.grid(row=3, column=0)
        label5.grid(row=4, column=0)

        callback_name_textbox.grid(row=0, column=1)
        flkernel_count_textbox.grid(row=1, column=1)
        batch_size_textbox.grid(row=2, column=1)
        epochs_textbox.grid(row=3, column=1)

        button_ok = Button(modal_frame, text='OK')
        button_ok.grid(row=4, column=1)
        resultcallback, resultkernel, resultbatch, resultepochs = '', '', '', ''

        def close_modal(event):
            nonlocal resultcallback, resultkernel, resultbatch, resultepochs

            resultcallback = str(callback_name_textbox.get())

            try:
                resultkernel = int(flkernel_count_textbox.get())
                resultbatch = int(batch_size_textbox.get())
                resultepochs = int(epochs_textbox.get())
                modal_window.destroy()
            except ValueError:
                label5.configure(text='Wrong Input!')

        button_ok.bind('<Button-1>', close_modal)

        modal_frame.pack()

        modal_window.wm_title("Neural network params")
        modal_window.focus_force()
        modal_window.geometry('300x125')

        modal_window.wait_window()

        return resultcallback, resultkernel, resultbatch, resultepochs

    def __predict_window(self):
        callback_name = StringVar()
        flkernel_count = StringVar()

        modal_window = Toplevel(self.root)

        modal_frame = Frame(modal_window)
        modal_window.resizable(width=False, height=False)

        label1 = Label(modal_frame, text='Callback name')
        label2 = Label(modal_frame, text='FL kernel count')
        label5 = Label(modal_frame, text='')

        dir = '../Unet/weights'
        files = os.listdir(dir)

        weights_name_list = Listbox(modal_frame, height=6, width=25)
        for i in files:
            weights_name_list.insert(0, i[:-5])
        flkernel_count_textbox = Entry(modal_frame, textvariable=flkernel_count, width=25)

        label1.grid(row=0, column=0)
        label2.grid(row=1, column=0)
        label5.grid(row=4, column=0)

        weights_name_list.grid(row=0, column=1)
        flkernel_count_textbox.grid(row=1, column=1)
        button_ok = Button(modal_frame, text='OK')
        button_ok.grid(row=4, column=0, columnspan=2)
        resultcallback, resultkernel, = '', ''

        def close_modal(event):
            nonlocal resultcallback, resultkernel

            resultcallback = str(weights_name_list.get(ACTIVE))

            try:
                resultkernel = int(flkernel_count_textbox.get())
                modal_window.destroy()
            except ValueError:
                label5.configure(text='Wrong Input!')

        button_ok.bind('<Button-1>', close_modal)

        modal_frame.pack()

        modal_window.wm_title("Neural network prediction params")
        modal_window.focus_force()
        modal_window.geometry('315x170')

        modal_window.wait_window()

        return resultcallback, resultkernel

    def train_network(self):
        callback, flkernel, batch, epochs = self.__train_window()
        self.network.train(callback, flkernel, batch, epochs)
        return 1

    def predict_test(self):
        callback, flkernel = self.__predict_window()
        self.network.predict(self.network.get_model_with_weights(callback, flkernel))

    def predict_intermediate(self):

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
        modal_window.resizable(width=False, height=False)

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
