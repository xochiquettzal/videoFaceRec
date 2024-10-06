import cv2

class VideoProcessor:
    def __init__(self, video_path, fps, skip_frames):
        self.video_path = video_path
        self.video_capture = cv2.VideoCapture(video_path)
        self.fps = fps
        self.skip_frames = skip_frames
        self.total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    def read_frame(self):
        ret, frame = self.video_capture.read()
        return ret, frame

    def release(self):
        self.video_capture.release()

    def get_remaining_time(self, processed_frames):
        remaining_frames = self.total_frames - processed_frames
        estimated_time_left = remaining_frames / self.fps
        return estimated_time_left
