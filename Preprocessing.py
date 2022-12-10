"データの水増しと整形並びに学習データの作成"
from PIL import Image
import os, glob
import numpy as np
from PIL import ImageFile
# IOError: image file is truncated (0 bytes not processed)回避のため
ImageFile.LOAD_TRUNCATED_IMAGES = True

# indexを教師ラベルとして割り当てるため、
# 0にはaragaki_outを指定し、1にはishihara_outを指定
# 2にはkitagawa_outを指定し、3にはnanao_outを指定
classes = ["kanna_out", "ishihara_out", "haru_out", "ebihara_out"]
num_classes = len(classes)
image_size = 64
#num_testdata = files1

X_train = []
X_test  = []
y_train = []
y_test  = []

#データ水増し & 訓練データ80% / テストデータ20% に分割
for index, classlabel in enumerate(classes):
    photos_dir = "./" + classlabel
    files = glob.glob(photos_dir + "/*.jpg")
    files1 = len(files)*0.2
    for i, file in enumerate(files):
        image = Image.open(file)
        image = image.convert("RGB")
        image = image.resize((image_size, image_size))
        #data = np.asarray(image)
        if i < files1:
            # angleに代入される値
            # -30, -25, -20 ... 20, 25, 30 と画像を5度ずつ回転
            for angle in range(-30, 30, 5):

                img_r = image.rotate(angle)
                data = np.asarray(img_r)
                X_test.append(data)
                y_test.append(index) # indexを教師ラベルとして割り当てるため、0にはCuteを、1にはElegantを、2にはCoolを、3にはFeminineを指定
                img_trains = img_r.transpose(Image.FLIP_LEFT_RIGHT) # FLIP_LEFT_RIGHT　は 左右反転
                data = np.asarray(img_trains)
                X_test.append(data)
                y_test.append(index) # indexを教師ラベルとして割り当てるため、0にはCuteを、1にはElegantを、2にはCoolを、3にはFeminineを指定
        else:

            # angleに代入される値
            # -30, -25, -20 ... 20, 25, 30 と画像を5度ずつ回転

            for angle in range(-30, 30, 5):

                img_r = image.rotate(angle)
                data = np.asarray(img_r)
                X_train.append(data)
                y_train.append(index) # indexを教師ラベルとして割り当てるため、0にはCuteを、1にはElegantを、2にはCoolを、3にはFeminineを指定
                img_trains = img_r.transpose(Image.FLIP_LEFT_RIGHT) # FLIP_LEFT_RIGHT　は 左右反転
                data = np.asarray(img_trains)
                X_train.append(data)
                y_train.append(index) # indexを教師ラベルとして割り当てるため、0にはdogを指定し、1には猫を指定

X_train = np.array(X_train)
X_test  = np.array(X_test)
y_train = np.array(y_train)
y_test  = np.array(y_test)

np.savez(r'C:\Users\toshi\Desktop\originalApp2\joyu4_random64_npz', X_train, X_test, y_train, y_test)


#xy = (X_train, X_test, y_train, y_test)
#np.save("./joyu4.npy", xy)