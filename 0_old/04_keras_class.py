# coding: utf-8

# In[1]:

import numpy as np
import cv2
import sys
from keras.models import load_model

# In[2]:

img_path = "./test/hayashi.jpg"

# In[3]:

answer = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E"}
img_rows, img_cols = 28, 28
model = load_model('05_model.h5')

# In[4]:

# 顔の最小値
MinSize = (30, 30)
# 画像読み込み
img_raw = cv2.imread(img_path)  # 画像を読み込む

# 画像縮小
height = img_raw.shape[0]
width = img_raw.shape[1]

re_width = int(500 / height * width)
re_height = 500

img = cv2.resize(img_raw, (re_width, re_height))
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # グレースケール化

# 顔認識(各画像スケールにおける縮小量,)
# cascade の学習結果は https://github.com/Itseez/opencv/tree/master/data/haarcascades から落とせる
cascade_path = "haarcascade_frontalface_alt.xml"
cascade = cv2.CascadeClassifier(cascade_path)  # カスケード分類器を作成
facerect = cascade.detectMultiScale(img_gray, scaleFactor=1.2, minNeighbors=1, minSize=MinSize)

test_image = []
# 顔だけ切り出して保存
for rect in facerect:
    x = rect[0]
    y = rect[1]
    width = rect[2]
    height = rect[3]
    # cut
    face = img[y:y + height, x:x + width]
    face = cv2.resize(face, (img_rows, img_cols))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    test_image.append(face.flatten().astype(np.float32) / 255.0)
test_image = 1 - np.asarray(test_image)
test_image = test_image.reshape(test_image.shape[0], img_rows, img_cols, 1)

# In[ ]:

pre = model.predict_classes(test_image)
print(pre)


# In[ ]:



