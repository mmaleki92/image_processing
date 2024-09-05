import sys
import cv2
import numpy as np
import sqlite3
from datetime import datetime
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget, QHBoxLayout, QMessageBox

class FaceRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # Initialize the face recognizer and face detector
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read('trained_face_model.yml')
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Set up the camera
        self.cap = cv2.VideoCapture(0)

        # Set up the SQLite database
        self.init_database()

        # To store detected IDs and their times
        self.id_times = {}

        # Set up the UI components
        self.initUI()

        # Set up a timer for updating the frame
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(20)  # Update frame every 20 milliseconds

    def init_database(self):
        # Connect to the SQLite database or create it if it doesn't exist
        self.conn = sqlite3.connect('face_recognition.db')
        self.cursor = self.conn.cursor()

        # Create a table for storing entry and exit logs
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER,
                entry_time TEXT,
                exit_time TEXT
            )
        ''')
        self.conn.commit()

    def initUI(self):
        # Main window properties
        self.setWindowTitle('Face Recognition')
        self.setGeometry(100, 100, 800, 600)

        # Create a central widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        # Video feed label
        self.video_label = QLabel()
        self.layout.addWidget(self.video_label)

        # Button layout for enter, exit, and quit
        button_layout = QHBoxLayout()

        # Enter button
        self.enter_button = QPushButton('Enter')
        self.enter_button.clicked.connect(self.record_entry)
        button_layout.addWidget(self.enter_button)

        # Exit button
        self.exit_button = QPushButton('Exit')
        self.exit_button.clicked.connect(self.record_exit)
        button_layout.addWidget(self.exit_button)

        # Quit button
        self.quit_button = QPushButton('Quit')
        self.quit_button.clicked.connect(self.close)
        button_layout.addWidget(self.quit_button)

        # Add button layout to the main layout
        self.layout.addLayout(button_layout)

    def update_frame(self):
        # Capture frame from the camera
        ret, frame = self.cap.read()
        if not ret:
            return

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        # Draw rectangles around faces and display labels
        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]

            # Predict the person
            label, confidence = self.recognizer.predict(face)
            print(f"Detected ID: {label} with Confidence: {confidence}")

            # Log detection time if ID is detected
            if label not in self.id_times:
                self.id_times[label] = {'entry': None, 'exit': None}

            # Display the label and rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, f"ID: {label}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Convert frame to RGB format and update the QLabel
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))

    def record_entry(self):
        # Record the entry time for detected IDs
        for label in self.id_times:
            if self.id_times[label]['entry'] is None:
                entry_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.id_times[label]['entry'] = entry_time
                self.cursor.execute("INSERT INTO logs (person_id, entry_time) VALUES (?, ?)", (label, entry_time))
                self.conn.commit()

                # Show a dialog box to confirm the entry
                self.show_message(f"ID: {label} entered at {entry_time}")

    def record_exit(self):
        # Record the exit time for detected IDs
        for label in self.id_times:
            if self.id_times[label]['entry'] is not None and self.id_times[label]['exit'] is None:
                exit_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.id_times[label]['exit'] = exit_time
                self.cursor.execute("UPDATE logs SET exit_time = ? WHERE person_id = ? AND exit_time IS NULL", (exit_time, label))
                self.conn.commit()

                # Show a dialog box to confirm the exit
                self.show_message(f"ID: {label} exited at {exit_time}")

    def show_message(self, message):
        # Display a message dialog box
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setText(message)
        msg_box.setWindowTitle("Information")
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()

    def closeEvent(self, event):
        # Release the camera, close the database connection, and close the application
        self.cap.release()
        self.conn.close()
        cv2.destroyAllWindows()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FaceRecognitionApp()
    window.show()
    sys.exit(app.exec_())
