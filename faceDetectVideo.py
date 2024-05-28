import cv2
import os

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cam = cv2.VideoCapture(0)

count = 0
while True:
    OK, frame = cam.read()
    faces = face_detector.detectMultiScale(frame, 1.3,5)

    for (x,y,w,h) in faces:
        face = cv2.resize(frame[y+2: y+h-2, x+2: x+w-2], (100,100))
        cv2.imwrite('faces_cam/face_{}.jpg'.format(count), face)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

        count += 1

    cv2.imshow('frame', frame)

    if count > 30:
        break
    elif cv2.waitKey(1) & 0xFF == ord('q'):
        break



cam.release()
cv2.destroyAllWindows()
