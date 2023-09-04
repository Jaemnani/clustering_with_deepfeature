import albumentations as A
import numpy as np
from glob import glob
from PIL import Image
import cv2
import os
from tqdm import tqdm

transform = A.Compose([
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Flip(p=0.5),
    A.RandomRotate90(p=0.5),
    # A.RandomBrightnessContrast(p=0.5, brightness_limit=0.3, contrast_limit=0.3),
    # A.GaussianBlur(p=0.5, blur_limit=(5,5)),
    # A.Rotate(limit=180, p=1, border_mode=cv2.BORDER_REPLICATE)
])
base_path = "./labeled/lens_samples/"

dst_path = "./labeled/augment/"

if not os.path.exists(dst_path):
    os.makedirs(dst_path)
image_list = glob(base_path + "/*.jpg")

for img_idx, img_path in enumerate(tqdm(image_list)):
    basename = os.path.basename(img_path)
    pillow_image = Image.open(img_path)
    image = np.array(pillow_image)
    cv2.imwrite(dst_path + basename, image)
    for i in range(499):
        dst_name = basename.replace(".jpg", "_%.3d.jpg"%(i))
        # dst_path = dst_path +"/"+ target+"/"+ dst_name
        transformed = transform(image=image)
        cv2.imwrite(dst_path + dst_name, transformed['image'])
        # print(dst_path+dst_name)
        

