from time import time
st = time()
from keras.applications.vgg19 import VGG19
import sys, os, numpy as np
from natsort import natsorted
from tqdm import tqdm
from glob2 import glob
import cv2

from LAMDA_SSL.Algorithm.Clustering.Constrained_Seed_k_means import Constrained_Seed_k_means
from LAMDA_SSL.Algorithm.Clustering.Constrained_k_means import Constrained_k_means

from sklearn.model_selection import train_test_split
import joblib
# np.set_printoptions(threshold=sys.maxsize)

tar_img_dir = './total/'
# train_img_folder = './labeled/augment/' # # 3000 ea
labeled_dir = './labeled/merge/' # # 3000 ea + clean data
unlabeled_dir = './unlabeled/lens_dirty/'

result_dir = "./result/"
k = 6 # clean, 1,2,3,4,5

#load model
base_model = None
base_model = VGG19(include_top=False, pooling='avg', weights='imagenet')


# # # Save File Check
labeled_filename = "./labeled_features.csv"
unlabeled_filename = './unlabeled_features.csv'
target_filename = "./target_features.csv"

if os.path.exists(labeled_filename):
    print(" Labeled Feature File Exists.")
    labeled_data = np.loadtxt(labeled_filename, dtype=str, delimiter=',')
    labeled_list = labeled_data[:,0]
    labels = labeled_data[:,1].astype(int)
    labeled_features = labeled_data[:, 2:].astype(float)
else:
    # preparing Image file for training
    labeled_list = glob(labeled_dir + "/*.jpg")
    labeled_list = natsorted(labeled_list)

    labels = np.zeros(len(labeled_list))
    for file_idx, file_name in enumerate(labeled_list):
        if "level_0" in file_name or "lens_clean" in file_name:
            labels[file_idx] = 0
            # print(file_name)
        elif "level_1" in file_name:
            labels[file_idx] = 1
        elif "level_2" in file_name:
            labels[file_idx] = 2
        elif "level_3" in file_name:
            labels[file_idx] = 3
        elif "level_4" in file_name:
            labels[file_idx] = 4
        elif "level_5" in file_name:
            labels[file_idx] = 5
        else:
            print('Error:', file_name)

    labeled_imgs = [cv2.imread(name) for name in labeled_list]
    labeled_imgs = [cv2.resize(img, (300,300)) for img in labeled_imgs]
    labeled_imgs = np.array(labeled_imgs)
    print(labeled_imgs.shape)

    #get feature 1
    labeled_features = base_model.predict(labeled_imgs)
    print(len(labeled_features))

    # Save feature with filename
    labeled_data = np.hstack((np.array(labeled_list).reshape(-1,1), labels.astype(int).reshape(-1,1), labeled_features)).astype('str')
    np.savetxt(labeled_filename, labeled_data, delimiter=",", fmt='%s')
    print("Saved", labeled_filename)

if os.path.exists(unlabeled_filename):
    print(" Unlabeled Feature File Exists.")
    unlabeled_data = np.loadtxt(unlabeled_filename, dtype=str, delimiter=',')
    unlabeled_list = unlabeled_data[:,0]
    unlabeled_features = unlabeled_data[:,1:].astype(float)
else:
    unlabeled_list = glob(unlabeled_dir + "/*.jpg")
    unlabeled_list = natsorted(unlabeled_list)

    unlabeled_imgs = [cv2.imread(name) for name in unlabeled_list]
    unlabeled_imgs = [cv2.resize(img, (300,300)) for img in unlabeled_imgs]
    unlabeled_imgs = np.array(unlabeled_imgs)
    print(unlabeled_imgs.shape)

    #get feature 2
    unlabeled_features = base_model.predict(unlabeled_imgs)
    print(len(unlabeled_features))

    # Save feature with filename
    unlabeled_data = np.hstack((np.array(unlabeled_list).reshape(-1,1), unlabeled_features)).astype('str')
    np.savetxt(unlabeled_filename, unlabeled_data, delimiter=",", fmt='%s')
    print('Saved',unlabeled_filename)

# from sklearn import preprocessing
# pre_transform = preprocessing.StandardScaler()
# pre_transform.fit(np.vstack((train_labeled_features, train_unlabeled_features)))
# train_labeled_features = pre_transform.transform(train_labeled_features)
# train_unlabeled_features = pre_transform.transform(train_unlabeled_features)
# print("preprocessing with sklearn")


x_lab_train, x_lab_test, y_lab_train, y_lab_test, list_lab_train, list_lab_test = train_test_split(labeled_features, labels, labeled_list, test_size=0.2, random_state=42)
x_unl_train, x_unl_test, list_unl_train, list_unl_test = train_test_split(unlabeled_features, unlabeled_list, test_size=0.5, random_state=42)


kmeans = Constrained_Seed_k_means(k=6)
kmeans.fit(X=x_lab_train, y=y_lab_train, unlabeled_X=x_unl_train)
print("Constrained Seed Kmeans Fitted")

# # Save Model
import joblib
joblib.dump(kmeans, "./test_save_kmeans.joblib")
# load_kmeans = joblib.load("./test_save_kmeans.joblib")

# save labels 1
# # sklearn kmeans
# train_save_labels = np.hstack((np.array(train_file_list_sorted).reshape(-1,1), np.array(kmeans.labels_).reshape(-1,1))).astype('str')
# # 
train_labels = np.hstack([list_lab_train, list_unl_train]).reshape(-1,1)
train_datas = np.hstack(( train_labels, np.array(kmeans.y).reshape(-1,1) )).astype('str')
np.savetxt('./train_datas.csv', train_datas, delimiter=",", fmt='%s')


if os.path.exists(target_filename):
    print(" Target Feature file is exists.")
    target_data = np.loadtxt(target_filename, dtype=str, delimiter=",")
    target_list = target_data[:,0]
    target_features = target_data[:,1:].astype(float)
else:
    # preparing Target Image file
    target_list = glob(tar_img_dir + "/*.jpg")
    target_list = natsorted(target_list)

    target_imgs = [cv2.imread(name) for name in target_list]
    target_imgs = [cv2.resize(img, (300,300)) for img in target_imgs]
    target_imgs = np.array(target_imgs)
    print(target_imgs.shape)

    # get feature 2
    target_features = base_model.predict(target_imgs).astype(float)
    print(len(target_features))

    # Save Feature With Filename
    target_data = np.hstack((np.array(target_list).reshape(-1,1), target_features)).astype('str')
    np.savetxt(target_filename, target_data, delimiter=",", fmt='%s')

pred = kmeans.predict(target_features, Transductive=False).astype(int)

#save labels 2
target_labels = np.hstack((np.array(target_list).reshape(-1,1), np.array(pred).reshape(-1,1))).astype('str')
np.savetxt('./target_labels.csv', target_labels, delimiter=',', fmt='%s')

ed = time()
print("process time : ", ed - st)

import shutil
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
print(' check Result Dir')

for filename, label in tqdm(zip(target_list, pred)):
    print(filename, label)
    if not os.path.exists(result_dir + "/" + str(label)):
        os.makedirs(result_dir +"/"+ str(label)+"/")
    shutil.copy( filename, result_dir+"/"+str(label)+"/")

print("done")