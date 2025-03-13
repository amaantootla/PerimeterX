import os, time, threading
from datetime import datetime
from pathlib import Path
import zmq


class SecurityServer:
    def __init__(self, model_port=5555, alert_port=5556, startup_delay=10):
        self.context = zmq.Context()
        self.model_port = model_port
        self.alert_port = alert_port
        self.running = False
        self.model_socket = None
        self.startup_delay = startup_delay
        

    def start(self):
        self.running = True
        self.model_thread = threading.Thread(target=self._serve_model)
        self.model_thread.daemon = True
        self.model_thread.start()
        
        self._handle_alerts()
        

    def _serve_model(self):
        self.model_socket = self.context.socket(zmq.PUB)
        self.model_socket.bind(f"tcp://*:{self.model_port}")
        
        print(f"\nWaiting {self.startup_delay} seconds for clients to connect...")
        for i in range(self.startup_delay, 0, -1):
            print(f"Starting in {i} seconds...", end='\r')
            time.sleep(1)
        print("\nSending initial model...")
        
        model_path = Path('database/home.pt')
        if model_path.exists():
            with open(model_path, 'rb') as f:
                model_data = f.read()
                self.model_socket.send_multipart([
                    b"model_update",
                    str(time.time()).encode(),
                    model_data
                ])
            print("Initial model sent successfully")
        
        model_path = Path('database/home.pt')
        last_modified = 0
        
        while self.running:
            try:
                if not model_path.exists():
                    time.sleep(10)
                    continue
                    
                current_modified = os.path.getmtime(model_path)
                if current_modified > last_modified:
                    print(f"[{datetime.now()}] Sending model update...")
                    with open(model_path, 'rb') as f:
                        model_data = f.read()
                        self.model_socket.send_multipart([
                            b"model_update",
                            str(current_modified).encode(),
                            model_data
                        ])
                    last_modified = current_modified
                    print("Model update sent successfully")
                    
                time.sleep(10)
                
            except Exception as e:
                print(f"Error in model distribution: {e}")
                time.sleep(5)
                
        self.model_socket.close()
        
        
    def _handle_alerts(self):
        socket = self.context.socket(zmq.REP)
        socket.bind(f"tcp://*:{self.alert_port}")
        
        print(f"[{datetime.now()}] Security server started")
        print(f"Model distribution on port {self.model_port}")
        print(f"Alert handling on port {self.alert_port}")
        
        while self.running:
            try:
                message = socket.recv_json()
                
                if message.get('type') == 'model_request':
                    print(f"\n📤 Model requested by device: {message.get('device_id')}")
                    model_path = Path('database/home.pt')
                    if model_path.exists():
                        with open(model_path, 'rb') as f:
                            model_data = f.read()
                            self.model_socket.send_multipart([
                                b"model_update",
                                str(time.time()).encode(),
                                model_data
                            ])
                    socket.send_json({
                        "status": "received",
                        "timestamp": str(datetime.now())
                    })
                    continue
                
                print("\n🚨 ALERT RECEIVED 🚨")
                print(f"Time: {datetime.now()}")
                print(f"Device ID: {message.get('device_id')}")
                print(f"Location: {message.get('location')}")
                print(f"Confidence: {message.get('confidence', 0):.2f}")
                
                socket.send_json({
                    "status": "received",
                    "timestamp": str(datetime.now())
                })
                
            except Exception as e:
                print(f"Error handling alert: {e}")
                try:
                    socket.send_json({"status": "error"})
                except:
                    pass
                    

    def stop(self):
        self.running = False
        if self.model_socket:
            self.model_socket.close()
        self.context.term()


if __name__ == "__main__":
    server = SecurityServer()
    try:
        server.start()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.stop()