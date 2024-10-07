import os
import sys
import time
import cv2
from cv2 import cuda
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

            # CUDA GpuMat'e yükle
            frame_cuda = cuda.GpuMat()
            frame_cuda.upload(frame)

            # Çözünürlük küçültme işlemini kaldırdık
            scales = [1.0, 0.75, 0.5, 0.25]
            detected = False

            for scale in scales:
                # Frame boyutunu yeniden boyutlandır
                scaled_frame_cuda = cuda.GpuMat()
                scaled_frame_cuda.upload(cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR))

                # GPU'dan indirerek numpy dizisine dönüştür
                scaled_frame_np = scaled_frame_cuda.download()

                # Yüzleri tespit et
                face_locations, matches = self.face_recognizer.compare_faces(scaled_frame_np)

                for match in matches:
                    if match:
                        time_in_video = frame_number / self.video_processor.fps
                        self.found_times.append(time_in_video)
                        print(f"Yüz bulundu! Zaman: {time_in_video:.2f} saniye")

                        original_size_frame_cuda = cuda.GpuMat()
                        original_size_frame_cuda.upload(cv2.resize(scaled_frame_np, (frame.shape[1], frame.shape[0]),interpolation=cv2.INTER_LINEAR))

                        # Gaussian blur ile düzeltmeyi kaldırdık
                        corrected_frame = original_size_frame_cuda.download()  # GpuMat'i numpy dizisine indiriyoruz

                        # Yüksek çözünürlüklü görseli kaydet
                        self.image_saver.save_image(corrected_frame, self.video_name, time_in_video)

                        frame_number += self.video_processor.skip_frames
                        self.video_processor.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                        detected = True
                        break

                if detected:
                    break

            processed_frames = frame_number + 1
            estimated_time_left = self.video_processor.get_remaining_time(processed_frames)

            hours = int(estimated_time_left // 3600)
            minutes = int((estimated_time_left % 3600) // 60)
            seconds = int(estimated_time_left % 60)

            sys.stdout.write(f"\rKalan süre (bitmeye): {hours}:{minutes:02}:{seconds:02}")
            sys.stdout.flush()

            frame_number += 1

        self.video_processor.release()
        self.display_found_times()

    def display_found_times(self):
        print("\nKişi şu zamanlarda bulundu:")
        for time in self.found_times:
            hours = int(time // 3600)
            minutes = int((time % 3600) // 60)
            seconds = int(time % 60)
            print(f"{hours}:{minutes:02}:{seconds:02}")
