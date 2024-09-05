import cv2
import os
import numpy as np

dataset_path = 'face_dataset'

# Initialize the face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Prepare training data
def prepare_training_data(dataset_path):
    faces = []
    labels = []
    label_dict = {}

    for image_name in os.listdir(dataset_path):
        image_path = os.path.join(dataset_path, image_name)
        gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Extract the person ID from the filename
        person_id = int(image_name.split('_')[0])
        label_dict[person_id] = person_id

        faces.append(gray_image)
        labels.append(person_id)

    return faces, np.array(labels), label_dict

faces, labels, label_dict = prepare_training_data(dataset_path)
recognizer.train(faces, labels)

# Save the trained model
recognizer.save('trained_face_model.yml')
print("Training complete. Model saved.")
