import cv2
import os
import numpy as np
from tensorflow.keras import models
from datetime import datetime

lstResult = [
    'Den Vau',
    'Hoang Huy',
    'Son Tung',
    'Toc Tien',
]

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

models = models.load_model('model-baocao_10epochs.keras')


# tạo hàm thêm thông tin vào file diemdanh
def diemDanh(name):
    # Kiểm tra xem file diemdanh.csv có tồn tại hay không
    if not os.path.exists("diemdanh.csv"):
        # Nếu không tồn tại, tạo file và ghi tiêu đề
        with open("diemdanh.csv", "w") as f:
            f.write("Name,Time\n")

    # Mở file để đọc và ghi
    with open("diemdanh.csv", "r+") as f:
        my_data_list = f.readlines()
        name_list = []
        for line in my_data_list:
            entry = line.strip().split(",")  # Sử dụng strip để loại bỏ khoảng trắng thừa
            name_list.append(entry[0])

        # Nếu tên chưa tồn tại trong danh sách, thêm vào file
        if name not in name_list:
            now = datetime.now()
            date_string = now.strftime("%H:%M:%S")
            f.write(f"{name},{date_string}\n")  # Sử dụng write thay vì writelines


cam = cv2.VideoCapture(0)

while True:
    OK, frame = cam.read()
    faces = face_detector.detectMultiScale(frame, 1.3, 5)

    for (x,y,w,h) in faces:
        face = cv2.resize(frame[y: y+h, x: x+w], (100, 100))

        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)  # Chuyển đổi ảnh sang grayscale

        face = face.reshape((-1, 100, 100, 1))  # Chuẩn hóa hình ảnh nếu cần
        predictions = models.predict(face)
        max_prob = np.max(predictions)
        result = np.argmax(predictions)
        print(result)
        print(max_prob)

        if max_prob < 0.5:  # Ngưỡng để xác định unknown
            label = "Unknown"
        else:
            label = lstResult[result]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(frame, label, (x + 15, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if label != "Unknown":
            diemDanh(label)

    cv2.imshow('FRAME', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()