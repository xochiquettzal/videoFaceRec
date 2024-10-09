import cv2
from deepface import DeepFace


class FaceRecognizer:
    def __init__(self, photo_path):
        print("Kişinin fotoğrafı yükleniyor...")

        # Yunet ONNX model yükle
        self.face_detector = cv2.FaceDetectorYN.create(
            "rsrc/lib/face_detection_yunet_2023mar.onnx",
            "",
            (320, 320),
            0.9,
            0.3,
            5000
        )

        # Load the reference image (photo_path)
        self.person_image = cv2.imread(photo_path)
        self.person_encoding = self.get_face_encoding(self.person_image)

    def get_face_encoding(self, image):
        # Yüz tespiti yap ve yüzün kodlamasını al
        h, w = image.shape[:2]
        self.face_detector.setInputSize((w, h))

        faces = self.face_detector.detect(image)[1]
        if faces is not None and len(faces) > 0:
            # Yunet detect() output might contain more than 5 values
            x, y, w, h = faces[0][:4].astype(int)  # Extract only x, y, width, and height
            face_roi = image[y:y + h, x:x + w]
            return face_roi  # Return the face region as the encoding for comparison
        return None

    def compare_faces(self, frame):

        # Yunet ile yüz tespiti yap
        h, w = frame.shape[:2]
        self.face_detector.setInputSize((w, h))

        face_locations = []
        matches = []

        faces = self.face_detector.detect(frame)[1]
        if faces is not None and len(faces) > 0:
            for face in faces:
                # Extract only the first four values (x, y, w, h)
                x, y, w, h = face[:4].astype(int)
                face_locations.append((y, x + w, y + h, x))

                # Extract face ROI (Region of Interest)
                face_roi = frame[y:y + h, x:x + w]

                # Compare the face encoding
                match = self.compare_face_encoding(face_roi)
                matches.append(match)

        return face_locations, matches

    def compare_face_encoding(self, face_roi):
        # Burada DeepFace kullanarak karşılaştırma yapıyoruz
        try:
            if face_roi is not None and self.person_encoding is not None:
                # Save the detected face to a temporary file for comparison
                cv2.imwrite("temp_face.jpg", face_roi)

                # Use DeepFace for face comparison
                result = DeepFace.verify(img1_path="temp_face.jpg", img2_path="rsrc/photos/person_photo.jpg",
                                         enforce_detection=False)

                # Return whether the faces match
                return result["verified"]
        except Exception as e:
            print(f"Karşılaştırma hatası: {e}")
        return False
