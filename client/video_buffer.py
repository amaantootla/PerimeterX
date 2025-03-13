import cv2
import numpy as np
from collections import deque
from datetime import datetime
import os

class VideoBuffer:
    def __init__(self, buffer_seconds=30, fps=30):
        """Initialize circular buffer for video frames"""
        self.buffer_size = buffer_seconds * fps
        self.buffer = deque(maxlen=self.buffer_size)
        self.fps = fps
        self.recording = False
        self.output_dir = "recordings"
        os.makedirs(self.output_dir, exist_ok=True)

    def add_frame(self, frame):
        """Add a frame to the circular buffer"""
        self.buffer.append({
            'frame': frame.copy(),
            'timestamp': datetime.now()
        })

    def save_buffer(self, seconds_before=15, seconds_after=15):
        """Start saving the buffer to disk"""
        if self.recording:
            return

        self.recording = True
        self.frames_before = int(seconds_before * self.fps)
        self.frames_after = int(seconds_after * self.fps)
        self.frames_to_save = list(self.buffer)[-self.frames_before:]
        self.continue_recording = True

    def add_recording_frame(self, frame):
        """Add frame while recording"""
        if not self.recording:
            return

        self.frames_to_save.append({
            'frame': frame.copy(),
            'timestamp': datetime.now()
        })

        if len(self.frames_to_save) >= self.frames_before + self.frames_after:
            self._save_video()
            self.recording = False

    def _save_video(self):
        """Save the buffered frames to a video file"""
        if not self.frames_to_save:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"incident_{timestamp}.mp4")

        height, width = self.frames_to_save[0]['frame'].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, self.fps, (width, height))

        for frame_data in self.frames_to_save:
            out.write(frame_data['frame'])

        out.release()
        print(f"Saved recording to {filename}")
        self.frames_to_save = []