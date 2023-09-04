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
target_img_folder = './total/'
train_img_folder = './labeled/augment/' # # 3000 ea
# train_img_folder = './labeled/merge/' # # 3000 ea + clean data
resorted_img_folder = "./result/vgg19_001/"
k = 6 # clean, 1,2,3,4,5

import shutil
if not os.path.exists(resorted_img_folder):
    os.makedirs(resorted_img_folder)

#load model
base_model = None
base_model = VGG19(include_top=False, pooling='avg', weights='imagenet')

# train_feature file check
if os.path.exists("./train_features.csv"):
    print(" Train Feature file exists.")
    train_read_data = np.loadtxt(resorted_img_folder+"/train_features.csv", dtype=str, delimiter=',')
    train_file_list_sorted = train_read_data[:, 0]
    train_features = train_read_data[:, 1:]


else:
    # preparing Image file for training
    train_file_list = glob(train_img_folder + "/*.jpg")
    train_file_list_sorted = natsorted(train_file_list)

    train_imgs = [cv2.imread(name) for name in train_file_list_sorted]
    train_imgs_resize = [cv2.resize(img, (300,300)) for img in train_imgs]
    train_imgs_array = np.array(train_imgs_resize)
    print(train_imgs_array.shape)

    #get feature 1
    train_features = base_model.predict(train_imgs_array)
    print(len(train_features))

    # Save feature with filename
    train_save_data = np.hstack((np.array(train_file_list_sorted).reshape(-1,1), train_features)).astype('str')
    np.savetxt(resorted_img_folder+"/train_features.csv", train_save_data, delimiter=",", fmt='%s')

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=6, random_state=0, n_init=10)
kmeans.fit(train_features)

# # Save Model
import joblib
joblib.dump(kmeans, "./clustering_dirty_kmeans.joblib")
# load_kmeans = joblib.load("./clustering_dirty_kmeans.joblib")


# save labels 1
train_save_labels = np.hstack((np.array(train_file_list_sorted).reshape(-1,1), np.array(kmeans.labels_).reshape(-1,1))).astype('str')
np.savetxt(resorted_img_folder+'/train_labels.csv', train_save_labels, delimiter=",", fmt='%s')


if os.path.exists(resorted_img_folder+"/target_features.csv"):
    print(" target Feature file exists.")
    train_read_data = np.loadtxt(resorted_img_folder+"/target_features.csv", dtype=str, delimiter=',')
    target_file_list_sorted = train_read_data[:, 0]
    target_features = train_read_data[:, 1:]
else:
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
    np.savetxt(resorted_img_folder+"/target_features.csv", target_save_data, delimiter=",", fmt='%s')

pred = kmeans.predict(target_features.astype(np.float32))

#save labels 2
target_save_labels = np.hstack((np.array(target_file_list_sorted).reshape(-1,1), np.array(pred).reshape(-1,1))).astype('str')
np.savetxt(resorted_img_folder+'/target_labels.csv', target_save_labels, delimiter=',', fmt='%s')

ed = time()
print("process time : ", ed - st)



for filename, label in tqdm(zip(target_file_list_sorted, pred)):
    # print(filename, label)
    if not os.path.exists(resorted_img_folder + "/" + str(label)):
        os.makedirs(resorted_img_folder +"/"+ str(label)+"/")
    shutil.copy( filename, resorted_img_folder+"/"+str(label)+"/")

print("done")