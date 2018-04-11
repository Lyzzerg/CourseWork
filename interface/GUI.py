from tkinter import *
from tkinter.messagebox import askyesno


class GUI:
    root = 'main window'
    main_menu = 'menu'
    first_menu = 'file menu'
    second_menu = 'filters menu'

    label = 'image before'
    label2 = 'image after'
    label3 = 'slices quantity'

    button_next = 'next slice'
    button_previous = 'previous slice'
    button_analyze = 'u-net'

    save_func = ''

    def __init__(self, root, open_func, save_func,
                 invert_filter, gradual_filter, logarithmic_filter, cancel_func,
                 next_button_func, previous_button_func, unet_enable_func, unet_disable_func, analyze_func):
        self.root = root
        self.root.wm_title('Tiff slices viewer')
        self.add_first_menu(open_func, save_func)
        self.add_second_menu(invert_filter, gradual_filter, logarithmic_filter, cancel_func)
        self.add_third_menu(unet_enable_func, unet_disable_func)
        self.add_buttons(next_button_func, previous_button_func)
        self.add_labels()
        self.save_func = save_func
        self.button_analyze = Button(text='Analyze')
        self.button_analyze.bind('<Button-1>', analyze_func)
        self.button_analyze.pack()

    def start(self):
        self.root.mainloop()

        return 1

    def add_labels(self):
        self.label = Label(self.root)
        self.label2 = Label(self.root)
        self.label3 = Label(self.root, text='Slice №\n1')

        self.label.pack(side=LEFT)
        self.label2.pack(side=RIGHT)
        self.label3.pack(side=TOP)

        return 1

    def add_buttons(self, next_button_func, previous_button_func):
        self.button_next = Button(text='Next Slice')
        self.button_previous = Button(text='Previous Slice')
        self.button_next.bind('<Button-1>', next_button_func)
        self.button_previous.bind('<Button-1>', previous_button_func)
        self.button_previous.pack(side=BOTTOM)
        self.button_next.pack(side=BOTTOM)

        return 1

    def add_first_menu(self, open_func, save_func):
        self.main_menu = Menu(self.root)
        self.root.config(menu=self.main_menu)
        self.first_menu = Menu(self.main_menu)

        self.main_menu.add_cascade(label='File', menu=self.first_menu)

        self.first_menu.add_command(label='Open', command=open_func)
        self.first_menu.add_command(label='Save as', command=save_func)
        self.first_menu.add_command(label='Exit', command=self.close_win)

        return 1

    def add_second_menu(self, invert_filter, gradual_filter, logarithmic_filter, cancel_func):
        self.second_menu = Menu(self.main_menu)

        self.main_menu.add_cascade(label='Edit', menu=self.second_menu)
        self.second_menu.add_command(label='inversion', command=invert_filter)
        self.second_menu.add_command(label='gradual', command=gradual_filter)
        self.second_menu.add_command(label='logarithmic', command=logarithmic_filter)
        self.second_menu.add_command(label='cancel changes', command=cancel_func)

        return 1

    def add_third_menu(self, unet_enable_func, unet_disable_func):
        self.second_menu = Menu(self.main_menu)

        self.main_menu.add_cascade(label='U-Net', menu=self.second_menu)
        self.second_menu.add_command(label='enable', command=unet_enable_func)
        self.second_menu.add_command(label='disable', command=unet_disable_func)
        return 1

    def change_first_label(self, new_image):
        self.label.configure(image=new_image)
        self.label.image = new_image

        return 1

    def change_second_label(self, new_image):
        self.label2.configure(image=new_image)
        self.label2.image = new_image

        return 1

    def change_third_label(self, new_num):
        self.label3.configure(text='Slice №\n' + str(new_num))

        return 1

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
