import cv2
import os

# Specify the path to the Haar Cascade XML file
haar_cascade_path = 'cascades/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_cascade_path)

# Path to the directory where the images will be stored
dataset_path = 'face_dataset'
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# Capture from default camera (webcam)
cap = cv2.VideoCapture(0)

# Define the unique identifier for each person
person_id = input("Enter unique ID for the person: ")
count = 0

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        face = gray[y:y + h, x:x + w]

        # Save the captured face image
        face_filename = os.path.join(dataset_path, f"{person_id}_{count}.jpg")
        cv2.imwrite(face_filename, face)

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.imshow('Face Enrollment', frame)

    # Break if 'q' is pressed or enough images have been captured
    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 30:
        break

cap.release()
cv2.destroyAllWindows()
