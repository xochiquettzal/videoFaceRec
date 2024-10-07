import face_recognition
import cv2
from cv2 import cuda

class FaceRecognizer:
    def __init__(self, photo_path):
        print("Kişinin fotoğrafı yükleniyor...")
        self.person_image = face_recognition.load_image_file(photo_path)
        self.person_encoding = face_recognition.face_encodings(self.person_image)[0]

    def compare_faces(self, frame):
        # Frame üzerinde yüz tespiti yapmadan önce numpy dizisine dönüştür
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        matches = []

        for face_encoding in face_encodings:
            match = face_recognition.compare_faces([self.person_encoding], face_encoding)
            matches.append(match[0])

        return face_locations, matches
