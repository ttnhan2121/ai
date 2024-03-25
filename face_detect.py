import cv2
import os

# Load pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# Function to detect faces in an image
def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return image


# Function to resize an image
def resize_image(image, width=None, height=None):
    if width is None and height is None:
        return image
    if width is None:
        ratio = height / image.shape[0]
        width = int(image.shape[1] * ratio)
    elif height is None:
        ratio = width / image.shape[1]
        height = int(image.shape[0] * ratio)
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized


# Function to compare faces
def compare_faces(image_to_compare, image_folder):
    image_to_compare_gray = cv2.cvtColor(image_to_compare, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(image_to_compare_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Load images from folder and resize them
    for filename in os.listdir(image_folder):
        path = os.path.join(image_folder, filename)
        img = cv2.imread(path)
        img_resized = resize_image(img, width=image_to_compare.shape[1], height=image_to_compare.shape[0])
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        for (x, y, w, h) in faces:
            roi = img_gray[y:y + h, x:x + w]
            result = cv2.matchTemplate(roi, image_to_compare_gray, cv2.TM_CCOEFF_NORMED)
            _, confidence, _, _ = cv2.minMaxLoc(result)
            print(f"Similarity with {filename}: {confidence}")
            if confidence > 0.8:  # Threshold for similarity
                cv2.imshow('Match', img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                return


# Open camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Perform face detection
    if ret:
        frame_with_faces = detect_faces(frame)

        # Compare faces with images in folder
        image_folder = "/Users/nhantran/Documents/AI/AI test/ai/image"
        compare_faces(frame, image_folder)

        # Display the result
        cv2.imshow('Face Detection', frame_with_faces)

    # Check for exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
