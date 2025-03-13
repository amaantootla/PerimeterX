from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
RECORDINGS_DIR = PROJECT_ROOT / "recordings"
DATABASE_DIR = PROJECT_ROOT / "database"

# Video settings
BUFFER_SECONDS = 30
CAMERA_FPS = 30
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# Network settings
DEFAULT_SERVER_ADDRESS = "localhost"
MODEL_PORT = 5555
ALERT_PORT = 5556
WEBRTC_PORT = 8080

# Model settings
PERSON_CONFIDENCE = 0.5
HOME_CONFIDENCE = 0.03