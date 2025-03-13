# PerimeterX: Smart Security System

A distributed AI-powered security system with real-time monitoring, adaptive learning, and smart home integration.

## Core Components

### Client-Server Architecture
- **Server**: Central hub for model training and distribution
- **Clients**: Edge devices (cameras) running inference
- Uses ZMQ (MQTT-like) for efficient pub/sub communication:
  - Model updates via PUB/SUB
  - Alerts via REQ/REP
  - Supports N edge devices with minimal overhead

### AI Model Pipeline
- Based on YOLOv11 nano for efficient edge inference
- Fine-tuning approach to prevent catastrophic forgetting:
  - Elastic Weight Consolidation (EWC) to preserve critical weights
  - Experience replay to maintain knowledge of previous patterns
  - Incremental learning for new household members
- Distributed model updates to edge devices

### Video Processing
- WebRTC streaming for real-time monitoring
- Circular buffer implementation:
  - 30-second rolling buffer
  - Captures 15s before and 15s after detection
  - Efficient bandwidth usage by only transmitting relevant segments
- CUDA acceleration where available

### Smart Home Integration
- OpenHAB integration hooks:
  - Trigger lights on detection
  - Activate alarms
  - Send notifications
  - Custom automation rules
- Extensible plugin architecture for other platforms

### Async & Parallel Processing
- Asyncio for non-blocking operations:
  - WebRTC streaming
  - Model inference
  - Alert handling
- Multi-threading for:
  - Video buffer management
  - Model updates
  - Real-time inference

## Installation

### Prerequisites
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install ultralytics opencv-python face-recognition pyzmq \
          aiohttp aiortc numpy requests pyyaml tqdm
```

### Server Setup
```bash
# Start the server
python -m server.security_server
```

### Client Setup
```bash
# Configure environment
export DEVICE_ID="camera_01"  # Unique ID for each camera
python -m client.security_client
```

## Project Structure
```
PerimeterX/
├── client/
│   ├── security_client.py   # Edge device main logic
│   ├── video_buffer.py      # Circular buffer implementation
│   └── webrtc_stream.py     # WebRTC streaming
├── server/
│   ├── training/
│   │   ├── trainer.py       # Model training pipeline
│   │   └── train.py        # Training CLI
│   └── security_server.py   # Central server logic
├── shared/
│   └── config.py           # Shared configurations
└── requirements.txt
```

## Key Features

### Real-time Detection
- Face recognition for known members
- Anomaly detection for unknown persons
- Low latency inference (<100ms)

### Adaptive Learning
- Continuous model improvement
- New member addition without forgetting
- Automatic model distribution

## Deployment

### Server Requirements
- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM
- Stable network connection

### Client Requirements
- Python 3.8+
- USB/IP camera
- 2GB+ RAM
- Network connectivity

## Contributing
1. Fork the repository
2. Create feature branch
3. Commit changes
4. Submit pull request