from time import time
st = time()
# from model import ImageCluster
from keras.applications.vgg19 import VGG19
import sys, os, numpy as np
from natsort import natsorted
from tqdm import tqdm
from glob2 import glob
import cv2
# np.set_printoptions(threshold=sys.maxsize)


# base_model = 'vgg19'
result_img_folder = '/home/jeremy/SynologyDrive/clobot/002_medinode_lens/classifier/dataset/broken/total/'
target_img_folder = '/home/jeremy/SynologyDrive/clobot/002_medinode_lens/classifier/dataset/broken/total/imgs/'

k = 6 # clean, 1,2,3,4,5

#load model
base_model = None
base_model = VGG19(include_top=False, pooling='avg', weights='imagenet')

# # Save Model

import joblib
# joblib.dump(kmeans, "./clustering_dirty_kmeans.joblib")
kmeans = joblib.load("./clustering_dirty_kmeans.joblib")

# preparing Target Image file
target_file_list = glob(target_img_folder + "/*.jpg")
target_file_list_sorted = natsorted(target_file_list)

target_imgs = [cv2.imread(name) for name in target_file_list_sorted]
target_imgs_resize = [cv2.resize(img, (300,300)) for img in target_imgs]
target_imgs_array = np.array(target_imgs_resize)
print(target_imgs_array.shape)

# get feature 2
target_features = base_model.predict(target_imgs_array).astype(float)
print(len(target_features))

# Save Feature With Filename
target_save_data = np.hstack((np.array(target_file_list_sorted).reshape(-1,1), target_features)).astype('str')
np.savetxt(result_img_folder+"/target_features.csv", target_save_data, delimiter=",", fmt='%s')

pred = kmeans.predict(target_features.astype(np.float32))

#save labels 2
target_save_labels = np.hstack((np.array(target_file_list_sorted).reshape(-1,1), np.array(pred).reshape(-1,1))).astype('str')
np.savetxt(result_img_folder+'/target_labels.csv', target_save_labels, delimiter=',', fmt='%s')

ed = time()
print("process time : ", ed - st)



for filename, label in tqdm(zip(target_file_list_sorted, pred)):
    # print(filename, label)
    if not os.path.exists(result_img_folder + "/" + str(label)):
        os.makedirs(result_img_folder +"/"+ str(label)+"/")
    import shutil
    shutil.copy( filename, result_img_folder+"/"+str(label)+"/")

print("done")