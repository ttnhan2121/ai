import cv2
import os

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def getFaces(image_path):
    img = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(img_gray, 1.3, 5)
    if len(faces) != 0:
        for (x, y, w, h) in faces:
            img_face = cv2.resize(img[y + 3:y + h - 3, x + 3:x + w - 3], (100, 100))
            dir_face_path = img_path.replace('image_raw', 'faces').split('.')[0]  # thay path cũ -> mới
            face_path = dir_face_path.split('\\')  # cắt thành path thư mục và tên

            cv2.imwrite(face_path[0] + '\\' + face_path[1] + '\\' + "hinh_{}.jpg".format(count), img_face)
            # print(face_path[0] + '\\' + face_path[1] + '\\' + "hinh_{}.jpg".format(count))
            return True
    return False


images_path = 'image_raw'

for dir in os.listdir(images_path):
    dir_path = os.path.join(images_path, dir)
    count = 0
    for sub_dir in os.listdir(dir_path):
        img_path = os.path.join(dir_path, sub_dir)

        if not os.path.isdir(dir_path.replace('image_raw', 'faces')):  # nếu không tồn tại dir faces
            os.mkdir(dir_path.replace('image_raw', 'faces'))

        if img_path.endswith(('.jpg', '.jpeg', '.png')):
            # print(img_path)
            if getFaces(img_path):
                count += 1



cv2.destroyAllWindows()
