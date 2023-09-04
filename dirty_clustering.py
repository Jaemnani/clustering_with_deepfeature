import matplotlib.pyplot as plt
import os
import shutil
from glob2 import glob
from sklearn.cluster import DBSCAN, HDBSCAN, OPTICS
import cv2
import numpy as np
from tqdm import tqdm

if not os.path.exists("./dst_clustering"):
    os.makedirs("./dst_clustering")

src_list = glob("./labeled/**/*.jpg")
dst_list = glob("./dataset/**/*.jpg")

src_labels_onehot = []
for i in src_list:
    if "level_5" in i:
        src_labels_onehot.append(np.array([0,0,0,0,0,1]))
    elif "level_4" in i:
        src_labels_onehot.append(np.array([0,0,0,0,1,0]))
    elif "level_3" in i:
        src_labels_onehot.append(np.array([0,0,0,1,0,0]))
    elif "level_2" in i:
        src_labels_onehot.append(np.array([0,0,1,0,0,0]))
    elif "level_1" in i:
        src_labels_onehot.append(np.array([0,1,0,0,0,0]))
    # elif "level_0" in i:
    else:
        src_labels_onehot.append(np.array([1,0,0,0,0,0]))

src_labels = []
for i in src_list:
    if "level_5" in i:
        src_labels.append(5.)
    elif "level_4" in i:
        src_labels.append(4.)
    elif "level_3" in i:
        src_labels.append(3.)
    elif "level_2" in i:
        src_labels.append(2.)
    elif "level_1" in i:
        src_labels.append(1.)
    # elif "level_0" in i:
    else:
        src_labels.append(0.)

src_imgs = [cv2.imread(name) for name in src_list]
src_resize = [cv2.resize(img, (354,354)) for img in src_imgs]
src_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in src_resize]
src_flatten = [img.astype(float).flatten() for img in src_gray]

x = 256
src_hist = [cv2.calcHist( images=[img], channels=[0], mask=None, histSize=[x], ranges=[0, 256]) for img in src_gray]
hist_flat = [img.flatten() for img in src_hist]
binx = np.arange(x) * 256/x

fft_img = np.fft.fftshift([np.fft.fft2(img, norm='ortho') for img in src_gray])
fft_magnitude_spectrum = 20 * np.log(np.abs(fft_img))
fft_flat = [np.abs(img).flatten() for img in fft_magnitude_spectrum]
DRAW = 0
if DRAW:
    for i, v in enumerate(tqdm(fft_img)):
        plt.title(src_list[i])
        plt.subplot(221), plt.imshow(src_gray[i], cmap='gray')
        plt.title('Input Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(222), plt.imshow(fft_magnitude_spectrum[i], cmap='gray')
        plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
        
        plt.subplot(223)
        plt.title('Histogram(%dd)'%(x))
        plt.plot(binx, hist_flat[i], color = 'r')
        plt.bar(binx, hist_flat[i], width=8, color='b')
        plt.ylim((0, 30_000))

        plt.show()

        plt.savefig("./figures/" + os.path.basename(src_list[i]).replace(".jpg", "_feature.png"))
        plt.cla()
        plt.clf()

total_feature = [np.append(f1, f2) for f1, f2 in zip(hist_flat, fft_flat)]

from sklearn.cross_decomposition import PLSRegression
# from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

print("start regression")
# 데이터 로드
# data = load_diabetes()
x = np.array(total_feature)
y = np.array(src_labels)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# PLS 모델 초기화
n_components = 4096  # 축소된 차원의 수
print("start regression instance")
pls = PLSRegression(n_components=n_components)

print("start regression fit")

# PLS 모델 학습
pls.fit(X_train, y_train)

# 학습된 PLS 모델을 사용하여 데이터 변환
X_train_pls = pls.transform(X_train)
X_test_pls = pls.transform(X_test)

print("start regression predict and error check")
# 예측 성능 측정
y_pred = pls.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")


from sklearn.cluster import kmeans_plusplus
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

test_feature = pls.transform(total_feature)
# centers, indices = kmeans_plusplus(test_feature, n_clusters=6, random_state=10)

kmeans = KMeans(n_clusters=6, random_state=10).fit(total_feature)



# dst_imgs = [cv2.imread(name) for name in dst_list]
print("done")