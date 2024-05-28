import numpy as np
import os
from PIL import Image
import tensorflow
from tensorflow.keras import layers
from tensorflow.keras import models


TRAIN_DATA = 'faces'

# 2 thư mục có các hình khác nhau

Xtrain = []  # mỗi phần từ là tuple (ma trận hình và nhãn)
Ytrain = []  # chứa nhãn tương ứng với ma trận hình

# tạo one-hot encoding
dict = {
    'denvau': [1,0,0,0],
    'huy': [0,1,0,0],
    'sontung': [0,0,1,0],
    'toctien': [0,0,0,1],
}

def getData(dirData, lstData):
    for face_dir in os.listdir(dirData):
        face_dir_path = os.path.join(dirData, face_dir)
        lst_face_path = []
        for face in os.listdir(face_dir_path):  # thêm tất cả hình của 1 thư mục vào
            face_path = os.path.join(face_dir_path, face)

            lable = face_path.split('\\')[1]  # lấy tên thư mục để tạo nhãn
            # img = np.array(Image.open(face_path))
            img = np.array(Image.open(face_path).convert('L'))  # Chuyển ảnh sang grayscale
            lst_face_path.append((img, dict[lable]))

        lstData.extend(lst_face_path)  # thêm mảng thư mục hình vào Xtrain
    return lstData

Xtrain = getData(TRAIN_DATA, Xtrain)

# print(Xtest[54])

# cifar10_cnn
model_training_first = models.Sequential([
    layers.Conv2D(32, (3,3), input_shape=(100,100,1), activation='relu'),
    layers.MaxPool2D((2,2)),
    layers.Dropout(0.15),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPool2D((2,2)),
    layers.Dropout(0,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPool2D((2,2)),
    layers.Dropout(0,2),

    layers.Flatten(),
    layers.Dense(1000, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(4, activation='softmax'),
])
model_training_first.summary()

model_training_first.compile(optimizer='adam',
                             loss='categorical_crossentropy',
                             metrics=['accuracy'])


# model_training_first.fit(np.array([x[0] for _, x in enumerate(Xtrain)]), np.array([y[1] for _, y in enumerate(Xtrain)]), epochs=10)
model_training_first.fit(np.array([x[0] for _, x in enumerate(Xtrain)]), np.array([y[1] for _, y in enumerate(Xtrain)]), batch_size=5, epochs=10)

model_training_first.save('model-baocao_10epochs.keras')



