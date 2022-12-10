"モデルの作成・学習・評価プログラム"
from tensorflow.keras import optimizers
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Dropout, Flatten, Input
from keras.models import Model, Sequential
from keras.utils.np_utils import to_categorical
import numpy as np

npz = np.load(r"C:\Users\toshi\Desktop\originalApp2/joyu4_random64_npz.npz")

X_train = npz['arr_0'] #訓練用画像データ
X_test = npz['arr_1']  #テスト用の画像データ
y_train = npz['arr_2'] #訓練用のラベルデータ
y_test = npz['arr_3']  #テスト用のラベルデータ

# 入力データの各画素値を0-1の範囲で正規化(学習コストを下げるため)
X_train = X_train.astype("float") / 255
X_test  = X_test.astype("float") / 255
# to_categorical()にてラベルをone hot vector化
y_train = to_categorical(y_train)
y_test  = to_categorical(y_test)

#X_train = npz_X[:int(len(npz_X)*0.8)]
#y_train = to_categorical(npz_y[:int(len(npz_y)*0.8)])
#x_test = npz_X[int(len(npz_X)*0.8):]
#y_test = to_categorical(npz_y[int(len(npz_y)*0.8):])

# ImageNetで事前学習した重みも読み込まれます
input_tensor = Input(shape=(64,64,3))
vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

# 新しく他の層を追加するために、あらかじめVGGとは別のモデル（ここではtop_model）を定義し、以下のようにして結合
top_model = Sequential()

top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(128, activation='relu'))
top_model.add(Dropout(0.5))
#top_model.add(Dense(64, activation='relu'))
#top_model.add(Dropout(0.25))
#top_model.add(Dense(32, activation='relu'))
top_model.add(Dense(4, activation='softmax'))
model=Model(inputs=vgg16.input,outputs=top_model(vgg16.output))

# modelの15層目までがvggのモデル
for layer in model.layers[:15]:
  layer.trainable = False

# loss = 損失関数 / optimizer = 最適化アルゴリズム / metrics = 評価関数
model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9), metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=15)

model.save('model_joyu4_random0000_npz.h5')

# モデルを評価する
scores = model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

model.summary()