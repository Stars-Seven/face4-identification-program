"顔の部分のみトリミングして保存するプログラム"
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import matplotlib.pyplot as plot
import os

# 入力ファイルのパスの指定
in_jpg = "haru" # 画像保存先フォルダ
out_jpg = "haru_out" # 顔切り抜き後の画像保存先フォルダ

if not os.path.isdir(out_jpg):
    # 顔部分をトリミングした後の画像保存先フォルダが存在しない場合、保存先フォルダを作成する
    os.makedirs(out_jpg)

# 保存している画像データの取得関数
def get_file(dir_path):
    filenames = os.listdir(dir_path)
    return filenames

pic = get_file(in_jpg)

for i in pic:
    # 画像の読み込み
    image_gs = cv2.imread(in_jpg + '/' + i)
    # 顔認識用特徴量ファイルを読み込む --- （カスケードファイルのパスを指定）
    cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    # 顔認識の実行
    face_list = cascade.detectMultiScale(image_gs,scaleFactor=1.1,minNeighbors=1,minSize=(100,100))

    # 顔だけ切り出して保存
    no = 0
    for rect in face_list:
        x = rect[0]
        y = rect[1]
        width = rect[2]
        height = rect[3]
        dst = image_gs[y:y + height, x:x + width]
        save_path = out_jpg + '/' + 'out_('  + str(i) +')' + str(no) + '.jpg'

        # 認識結果の保存
        a = cv2.imwrite(save_path, dst)
        no += 1