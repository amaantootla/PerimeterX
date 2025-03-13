import os, time, threading, requests
from datetime import datetime
from pathlib import Path
import cv2, zmq
from ultralytics import YOLO
from .webrtc_stream import BufferedVideoStreamTrack


class SecurityClient:
    def __init__(self, server_address="localhost", model_port=5555, alert_port=5556):
        self.context = zmq.Context()
        self.server_address = server_address
        self.model_port = model_port
        self.alert_port = alert_port
        self.running = False
        self.model_path = Path('home.pt')
        self.video_stream = BufferedVideoStreamTrack()
        
        
    def start(self):
        self.running = True
        self.model_thread = threading.Thread(target=self._receive_model)
        self.model_thread.daemon = True
        self.model_thread.start()
        
        self.inference_thread = threading.Thread(target=self.run_inference)
        self.inference_thread.daemon = True
        self.inference_thread.start()
        
        print(f"Connected to server at {self.server_address}")
        print(f"Waiting for model update...")
        

    def request_initial_model(self):
        print("Requesting initial model from server...")
        socket = self.context.socket(zmq.REQ)
        socket.connect(f"tcp://{self.server_address}:{self.alert_port}")
        
        try:
            alert = {
                "device_id": os.getenv("DEVICE_ID", "unknown_device"),
                "type": "model_request",
                "timestamp": str(datetime.now())
            }
            socket.send_json(alert)
            
            response = socket.recv_json()
            if response['status'] == 'received':
                print("Server will send model...")
            else:
                print("Failed to request model")
                
        except Exception as e:
            print(f"Error requesting model: {e}")
        finally:
            socket.close()
        

    def _receive_model(self):
        socket = self.context.socket(zmq.SUB)
        socket.connect(f"tcp://{self.server_address}:{self.model_port}")
        socket.setsockopt(zmq.SUBSCRIBE, b"model_update") 
        print("Waiting for model updates...")
        
        while self.running:
            try:
                if socket.poll(timeout=1000):
                    parts = socket.recv_multipart()
                    if len(parts) == 3:
                        topic, timestamp, model_data = parts
                        print(f"\n📥 Receiving model update ({len(model_data)} bytes)")
                        
                        with open(self.model_path, 'wb') as f:
                            f.write(model_data)
                        print(f"✅ Model saved as {self.model_path}")
                    else:
                        print(f"❌ Received malformed message: {len(parts)} parts")
                
            except Exception as e:
                print(f"❌ Error receiving model: {e}")
                time.sleep(1)
                continue
                
        socket.close()
        

    def send_alert(self, location, confidence):
        socket = self.context.socket(zmq.REQ)
        socket.connect(f"tcp://{self.server_address}:{self.alert_port}")
        
        try:
            alert = {
                "device_id": os.getenv("DEVICE_ID", "unknown_device"),
                "location": location,
                "confidence": confidence,
                "timestamp": str(datetime.now())
            }
            
            print("\n🚨 Sending alert to server...")
            socket.send_json(alert)
            
            response = socket.recv_json()
            print(f"Server response: {response['status']}")
            
        except Exception as e:
            print(f"Error sending alert: {e}")
        finally:
            socket.close()
            

    def stop(self):
        self.running = False
        self.context.term()
        

    def download_base_model(self):
        base_model_path = 'yolo11n.pt'
        if not Path(base_model_path).exists():
            print("⏳ Downloading base model...")
            url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
            response = requests.get(url, stream=True)
            with open(base_model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("✅ Base model downloaded successfully")


    def run_inference(self, person_conf=0.5, home_conf=0.03):
        print("Waiting for home.pt model to be available...")
        while not Path('home.pt').exists():
            time.sleep(1)
        
        print("Loading models...")
        try:
            self.download_base_model()
            
            base_model = YOLO('yolo11n.pt')
            home_model = YOLO('home.pt')
            
            print("Starting inference...")
            while self.running:
                frame = self.video_stream.get_current_frame()
                if frame is None:
                    time.sleep(0.1)
                    continue
                
                person_results = base_model.predict(frame, classes=[0], conf=person_conf, verbose=False)
                if len(person_results[0].boxes) > 0:
                    home_results = home_model.predict(frame, conf=home_conf, verbose=False)
                    
                    if len(person_results[0].boxes) > len(home_results[0].boxes):
                        confidence = 1.0 - (len(home_results[0].boxes) / len(person_results[0].boxes))
                        self.video_stream.start_recording()
                        self.send_alert("Camera Feed", confidence)
                        time.sleep(10)
                
                time.sleep(0.1)
            
        except Exception as e:
            print(f"Error in inference: {e}")
            time.sleep(5)
            self.run_inference(person_conf, home_conf)


if __name__ == "__main__":
    client = SecurityClient()
    try:
        client.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down client...")
        client.stop()