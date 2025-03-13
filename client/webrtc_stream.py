import asyncio
import cv2
from av import VideoFrame
from aiortc import MediaStreamTrack
from .video_buffer import VideoBuffer

class BufferedVideoStreamTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, camera_id=0, buffer_seconds=30):
        super().__init__()
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.buffer = VideoBuffer(buffer_seconds=buffer_seconds)
        self._current_frame = None

    async def recv(self):
        frame = await self._capture_frame()
        return await self._process_frame(frame)

    async def _capture_frame(self):
        while True:
            ret, frame = self.cap.read()
            if ret:
                self._current_frame = frame.copy()
                self.buffer.add_frame(frame)
                if self.buffer.recording:
                    self.buffer.add_recording_frame(frame)
                return frame
            await asyncio.sleep(0.01)

    async def _process_frame(self, frame):
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1)

        # Create VideoFrame for WebRTC
        video_frame = VideoFrame.from_ndarray(frame, format="rgb24")
        pts, time_base = await self.next_timestamp()
        video_frame.pts = pts
        video_frame.time_base = time_base

        return video_frame

    def start_recording(self):
        """Start recording buffer when intruder detected"""
        self.buffer.save_buffer()

    def get_current_frame(self):
        """Get current frame for inference"""
        return self._current_frame