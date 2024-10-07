from face_detection_app import FaceDetectionApp

if __name__ == "__main__":
    photo_path = "rsrc/photos/person_photo.jpg"
    video_path = "rsrc/videos/kotuadamin10gunu.mp4"
    app = FaceDetectionApp(photo_path, video_path)
    app.process_video()
