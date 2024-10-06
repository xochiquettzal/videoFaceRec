import os
import sys
import time
import cv2
from face_recognizer import FaceRecognizer
from video_processor import VideoProcessor
from image_saver import ImageSaver


class FaceDetectionApp:
    def __init__(self, photo_path, video_path):
        self.face_recognizer = FaceRecognizer(photo_path)
        self.video_name = os.path.splitext(os.path.basename(video_path))[0]
        self.output_dir = f"rsrc/founds/{self.video_name}/"
        self.image_saver = ImageSaver(self.output_dir)
        self.video_processor = VideoProcessor(video_path, cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS), 50)
        self.found_times = []

    def process_video(self):
        frame_number = 0
        start_time = time.time()

        print("Yüzler inceleniyor...")
        while self.video_processor.video_capture.isOpened():
            ret, frame = self.video_processor.read_frame()

            if not ret:
                break

            if frame_number % 50 != 0:
                frame_number += 1
                continue

            # Çözünürlüğü çeyreğe indir
            frame = cv2.resize(frame, (frame.shape[1] // 8, frame.shape[0] // 8))

            face_locations, matches = self.face_recognizer.compare_faces(frame)

            for match in matches:
                if match:
                    time_in_video = frame_number / self.video_processor.fps
                    self.found_times.append(time_in_video)
                    print(f"Yüz bulundu! Zaman: {time_in_video:.2f} saniye")

                    # 4x upscale yap ve piksel düzeltme
                    upscaled_frame = cv2.resize(frame, None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)

                    # Yüzleri pikselleri düzeltme işlemi
                    # Burada bir işlem uygulamak için bir fonksiyon tanımlamanız gerekebilir
                    corrected_frame = self.correct_pixels(upscaled_frame)

                    self.image_saver.save_image(corrected_frame, self.video_name, time_in_video)

                    frame_number += self.video_processor.skip_frames
                    self.video_processor.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                    break

            processed_frames = frame_number + 1
            estimated_time_left = self.video_processor.get_remaining_time(processed_frames)

            # Saat:Dakika:Saniye formatına dönüştür
            hours = int(estimated_time_left // 3600)
            minutes = int((estimated_time_left % 3600) // 60)
            seconds = int(estimated_time_left % 60)

            sys.stdout.write(f"\rKalan süre (bitmeye): {hours}:{minutes:02}:{seconds:02}")
            sys.stdout.flush()

            frame_number += 1

        self.video_processor.release()
        self.display_found_times()

    def correct_pixels(self, frame):
        # Burada pikselleri düzeltmek için uygun bir yöntem uygulayın.
        # Örneğin, gauss bulanıklığı veya başka bir filtre uygulanabilir.
        return cv2.GaussianBlur(frame, (5, 5), 0)

    def display_found_times(self):
        print("\nKişi şu zamanlarda bulundu:")
        for time in self.found_times:
            hours = int(time // 3600)
            minutes = int((time % 3600) // 60)
            seconds = int(time % 60)
            print(f"{hours}:{minutes:02}:{seconds:02} ")
