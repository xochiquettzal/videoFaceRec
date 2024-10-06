import cv2
import os

class ImageSaver:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def save_image(self, image, video_name, time_in_video):
        hours = int(time_in_video // 3600)
        minutes = int((time_in_video % 3600) // 60)
        seconds = int(time_in_video % 60)
        formatted_time = f"{hours:02}_{minutes:02}_{seconds:02}"
        output_path = os.path.join(self.output_dir, f"{video_name}_{formatted_time}.jpg")
        cv2.imwrite(output_path, image)
