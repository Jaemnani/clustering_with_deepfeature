import os
import cv2
import numpy as np
from glob2 import glob
import tkinter
from tkinter import filedialog

global_windows_name = "dirtyLabel"
global_image_size = (512,512)

class LabelDirty:
    def __init__(self):
        self.image_list = self.init_image_list()
        self.cur_image_path = ""
        self.show_label = True
        self.cur_img = None
        self.reset_label()

    def init_image_list(self):
        import natsort
        import tkinter
        from glob2 import glob
        from tkinter import filedialog
        root = tkinter.Tk()
        root.withdraw()
        root.wm_attributes("-topmost", 1)
        image_path = filedialog.askdirectory()
        print(image_path)
        image_list = natsort.natsorted(glob(image_path+"/**/*.jpg"))
        image_list = [name.replace("\\", "/") for name in image_list]
        return image_list
    
    def load_label_if_exists(self):
        label_path = self.cur_label_path
        if os.path.exists(label_path) and os.path.isfile(label_path):
            print("file exists")
            with open(label_path, 'r') as f:
                data = f.readlines()
                data = data[0]
                self.pre_label = int(data)
                self.cur_label = int(data)
        else:
            self.reset_label()
            print("no file")

    def update(self):
        img = self.img.copy()
        if self.show_label:

            font_size = 2
            cv2.putText(img, str(self.cur_label), (10,font_size *10 +10), 1, font_size, (255,255,255), 2)
        cv2.imshow(global_windows_name, img)

    def save_label(self):
        if self.pre_label != self.cur_label:
            if self.cur_label == -1:
                if os.path.exists(self.cur_label_path) and os.path.isfile(self.cur_label_path):
                    os.remove(self.cur_label_path)
            else:
                with open(self.cur_label_path, 'w') as f:
                    f.write(str(self.cur_label))
        

    def reset_label(self):
        self.pre_label = -1
        self.cur_label = -1
        
    def find_unlabeled_image_index(self):
        print('start to find unlabeled file')
        for img_idx, img_name in enumerate(self.image_list):
            label_name = img_name.replace(".jpg", ".txt")
            if os.path.exists(label_name) and os.path.isfile(label_name):
                continue
            else:
                return img_idx
        return -1

    def run(self):
        index = 0
        cv2.namedWindow(global_windows_name)
        while True:
            self.cur_image_path = self.image_list[index]
            self.cur_label_path = self.cur_image_path.replace(".jpg", ".txt")
            self.img = cv2.resize(cv2.imread(self.cur_image_path), global_image_size )
            self.load_label_if_exists()
            self.update()
            while True:
                res = cv2.waitKey(0)
                # print("Keyboard down : ", str(res))
                # continue
                if res == ord('d'):  # go to next if input key was 'd'
                    self.save_label()
                    if index == len(self.image_list) - 1:
                        print('Current image is last image')
                    else:
                        # self.limb_index = 0
                        # self.cur_label = -1
                        index += 1
                        break
                elif res == ord('a'):  # go to previous image if input key was 'a'
                    self.save_label()
                    if index == 0:
                        print('Current image is first image')
                    else:
                        # self.limb_index = 0
                        # self.cur_label = -1
                        index -= 1
                        break
                elif res == ord('w'):  # toggle show skeleton
                    self.show_label = not self.show_label
                    break
                elif res == ord("1"):
                    # print("1")
                    self.cur_label = 1
                    self.update()
                elif res == ord("2"):
                    self.cur_label = 2
                    self.update()
                elif res == ord("3"):
                    self.cur_label = 3
                    self.update()
                elif res == ord("4"):
                    self.cur_label = 4
                    self.update()
                elif res == ord("5"):
                    self.cur_label = 5
                    self.update()
                elif res == ord("0"):
                    self.cut_label = 0
                    self.update()
                elif res == ord('f'):  # auto find not labeled image
                    not_labeled_index = self.find_unlabeled_image_index()
                    if not_labeled_index != -1:
                        index = not_labeled_index
                        break
                elif res == ord('x'):  # remove cur label
                    self.cur_label = -1
                    # self.save_label()
                    # break
                elif res == ord('u'):
                    self.cur_label = self.pre_label
                    # break
                elif res == 27:  # exit if input key was ESC
                    self.save_label()
                    exit(0)

if __name__ == '__main__':
    LabelDirty().run()
print("done")


