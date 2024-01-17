import os
from tkinter import Label, Button
from PIL import Image, ImageTk


class ImageSlider:
    def __init__(self, root, image_folder):
        """
            for using in another file do:
            imports:
                1. from tkinter import Toplevel
                2. from image_slider import ImageSlider
            create next:
                1. root = Toplevel()
                2. root.title("Image Slider")
                3. root.protocol("WM_DELETE_WINDOW", root.destroy)
                4. image_folder_filepath = "all_image_plots"
                5. slider = ImageSlider(root, image_folder_filepath)
                6. root.mainloop()
        """
        self.root = root
        self.image_folder = image_folder
        self.image_list = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
        self.current_index = 0

        self.image_label = Label(root)
        self.image_label.pack()

        self.prev_button = Button(root, text="Previous plot", command=self.show_prev_image)
        self.prev_button.pack(side="left")

        self.next_button = Button(root, text="Next plot", command=self.show_next_image)
        self.next_button.pack(side="right")

        self.photo = None  # Добавим переменную для хранения ссылки на объект PhotoImage

        self.show_image()

    def show_image(self):
        image_path = os.path.join(self.image_folder, self.image_list[self.current_index])
        image = Image.open(image_path)
        self.photo = ImageTk.PhotoImage(image)
        self.image_label.config(image=self.photo)

    def show_next_image(self):
        self.current_index = (self.current_index + 1) % len(self.image_list)
        self.show_image()

    def show_prev_image(self):
        self.current_index = (self.current_index - 1) % len(self.image_list)
        self.show_image()

    def update_image(self):
        image_path = os.path.join(self.image_folder, self.image_list[self.current_index])
        image = Image.open(image_path)
        self.photo = ImageTk.PhotoImage(image)
        self.image_label.config(image=self.photo)
        self.image_label.image = self.photo
