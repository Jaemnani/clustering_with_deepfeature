# # + dirty clean labeling in filename as folder_name

import os
import shutil
import cv2
from glob import glob
import numpy as np
from tqdm import tqdm

file_list1 = np.array(glob("./dataset/**/*.jpg"))
file_list2 = np.array(glob("./dataset/**/**/*.jpg"))
file_list = np.concatenate((file_list1, file_list2))

# img_list = [cv2.imread(name) for name in file_list]
# crop_list = [img[:, int(img.shape[1]/2):] for img in img_list]
# save_img = [cv2.imwrite(img, name) for img, name in zip(crop_list, file_list)]

for file_idx, file_name in enumerate(tqdm(file_list)):
    img = cv2.imread(file_name)
    if img.shape[0] == img.shape[1]:
        # print(file_name, img.shape)
        if "lens_clean" in file_name:
            if not "_c.jpg" in file_name:
                dst_path = file_name.replace(".jpg", "_c.jpg")
                shutil.move(file_name, dst_path)
            # print('clean')
        elif "lens_dirty" in file_name:
            if not "_d.jpg" in file_name:
                dst_path = file_name.replace(".jpg", "_d.jpg")
                shutil.move(file_name, dst_path)
            # print("dirty")
        else:
            print(file_name)
        continue
    crop = img[:, int(img.shape[1]/2):, :]
    dst_path = file_name.replace(".jpg", "_R.jpg")
    cv2.imwrite(dst_path, crop)
    os.remove(file_name)    
    # print(dst_path, crop.shape)

print('done')

