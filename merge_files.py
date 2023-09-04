import os
import shutil
from glob2 import glob
from natsort import natsorted

src_path_non_labeled = "./dataset/"
src_path_labeled = "./labeled/"
file_list_non_labeled = glob( src_path_non_labeled + "**/*.jpg")
file_list_labeled = glob( src_path_labeled + "**/*.jpg")
base_name_non_labeled = [name.replace(src_path_non_labeled, "").replace("/", "_") for name in file_list_non_labeled]
base_name_labeled = [os.path.basename(name) for name in file_list_labeled]

dst_path = "./total/"
dst_name_non_labeled = [dst_path+name for name in base_name_non_labeled]
dst_name_labeled = [dst_path+name for name in base_name_labeled]

if not os.path.exists(dst_path):
    os.makedirs(dst_path)

[shutil.copy(src=src, dst=dst) for src, dst in zip(file_list_non_labeled, dst_name_non_labeled)]
[shutil.copy(src=src, dst=dst) for src, dst in zip(file_list_labeled, dst_name_labeled)]
print('done')