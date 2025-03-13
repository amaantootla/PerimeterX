import asyncio
import json
import cv2
import numpy as np
from aiohttp import web
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from av import VideoFrame

class VideoStreamTrack(MediaStreamTrack):
    kind = "video"
    
    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    async def recv(self):
        # Loop until a frame is captured
        while True:
            ret, frame = self.cap.read()
            if ret:
                break
            await asyncio.sleep(0.01)
        
        # Process the captured frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1)  # Mirror effect
        
        # Simulate frame rate of ~30 FPS
        await asyncio.sleep(1/30)
        
        video_frame = VideoFrame.from_ndarray(frame, format="rgb24")
        pts, time_base = await self.next_timestamp()
        video_frame.pts = pts
        video_frame.time_base = time_base
        
        return video_frame

async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    
    pc = RTCPeerConnection()
    pcs.add(pc)  # Keep track of peer connections
    
    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)
    
    # Add video track
    pc.addTrack(VideoStreamTrack())
    
    # Handle offer
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    
    return web.Response(
        content_type="application/json",
        text=json.dumps({
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        })
    )

async def on_shutdown(app):
    # Close all peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

if __name__ == "__main__":
    pcs = set()  # Global variable to store peer connections
    
    # Create web application
    app = web.Application()
    app.router.add_post("/offer", offer)
    app.router.add_get("/", lambda r: web.FileResponse("index.html"))
    app.on_shutdown.append(on_shutdown)
    
    # Start server
    web.run_app(app, host="0.0.0.0", port=8080)