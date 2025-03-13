import os
import cv2
import random
import face_recognition
import requests
from pathlib import Path
from ultralytics import YOLO
from ..shared.config import DATABASE_DIR

class SecurityModelTrainer:
    def __init__(self):
        self.database_dir = DATABASE_DIR
        self.train_dir = self.database_dir / 'train'
        self.val_dir = self.database_dir / 'val'
        self.model_path = self.database_dir / 'yolo11n.pt'
        self.final_model_path = self.database_dir / 'home.pt'
        
    def setup_directories(self):
        """Create necessary directories for training data"""
        for split in ['train', 'val']:
            (self.database_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.database_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
            (self.database_dir / split / 'annotated').mkdir(parents=True, exist_ok=True)

    def capture_images(self, split_ratio=0.8, source=0):
        """Capture and save training images from camera"""
        cap = cv2.VideoCapture(source)
        image_count = {'train': 0, 'val': 0}
        total_count = 0

        print("Press 'c' to capture an image, 'q' to quit")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print(f"Error: Could not access camera {source}")
                break
            
            cv2.imshow("Preview", frame)
            key = cv2.waitKey(1)
            
            if key == ord('c'):
                split = 'train' if random.random() < split_ratio else 'val'
                img_path = self.database_dir / split / 'images' / f'image_{total_count}.jpg'
                cv2.imwrite(str(img_path), frame)
                
                image_count[split] += 1
                total_count += 1
                
                print(f"Saved: {img_path} (Train: {image_count['train']}, Val: {image_count['val']})")
            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def process_images(self):
        """Process captured images and create YOLO format labels"""
        for split in ['train', 'val']:
            images_dir = self.database_dir / split / 'images'
            labels_dir = self.database_dir / split / 'labels'
            annotated_dir = self.database_dir / split / 'annotated'

            for img_path in images_dir.glob('*.jpg'):
                image = cv2.imread(str(img_path))
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                face_locations = face_recognition.face_locations(rgb_image)
                if not face_locations:
                    continue

                height, width = image.shape[:2]
                bbox = self._convert_to_yolo_bbox(face_locations[0], width, height)
                
                label_path = labels_dir / f"{img_path.stem}.txt"
                with open(label_path, 'w') as f:
                    f.write(f"0 {' '.join(map(str, bbox))}\n")
                
                self._draw_bbox(image, face_locations[0])
                cv2.imwrite(str(annotated_dir / img_path.name), image)

    def _convert_to_yolo_bbox(self, face_location, img_width, img_height):
        """Convert face location to YOLO format"""
        top, right, bottom, left = face_location
        x_center = (left + right) / 2.0 / img_width
        y_center = (top + bottom) / 2.0 / img_height
        width = (right - left) / img_width
        height = (bottom - top) / img_height
        return (x_center, y_center, width, height)

    def _draw_bbox(self, image, face_location):
        """Draw bounding box on image"""
        top, right, bottom, left = face_location
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

    def verify_labels(self):
        """Verify generated labels visually"""
        for split in ['train', 'val']:
            images_dir = self.database_dir / split / 'images'
            labels_dir = self.database_dir / split / 'labels'

            for img_path in images_dir.glob('*.jpg'):
                label_path = labels_dir / f"{img_path.stem}.txt"
                if not label_path.exists():
                    print(f"Warning: No label for {img_path.name}")
                    continue

                image = cv2.imread(str(img_path))
                height, width = image.shape[:2]
                
                with open(label_path) as f:
                    line = f.readline().strip()
                    _, *bbox = map(float, line.split())
                    self._draw_yolo_bbox(image, bbox, width, height)

                cv2.imshow(f'Verification ({split})', image)
                key = cv2.waitKey(500)
                if key == ord('q'):
                    break

        cv2.destroyAllWindows()

    def _draw_yolo_bbox(self, image, bbox, width, height):
        """Draw YOLO format bounding box on image"""
        x_center, y_center, w, h = bbox
        x = int((x_center - w/2) * width)
        y = int((y_center - h/2) * height)
        w = int(w * width)
        h = int(h * height)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    def train_model(self):
        """Train the YOLO model"""
        if not self.model_path.exists():
            self._download_base_model()
            
        data_yaml = self._create_dataset_yaml()
        model = YOLO(str(self.model_path))
        model.train(
            data=data_yaml,
            epochs=50,
            imgsz=640,
            project=str(self.database_dir / 'runs'),
            name='detect/train'
        )
        
        self._save_best_model()

    def _download_base_model(self):
        """Download pre-trained YOLO model"""
        url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
        print(f"Downloading base model from {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(self.model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Base model downloaded successfully")

    def _create_dataset_yaml(self):
        """Create YAML configuration for training"""
        yaml_path = self.database_dir / 'dataset.yaml'
        data = {
            'path': str(self.database_dir),
            'train': str(self.train_dir / 'images'),
            'val': str(self.val_dir / 'images'),
            'nc': 1,
            'names': ['home']
        }
        
        with open(yaml_path, 'w') as f:
            for key, value in data.items():
                f.write(f"{key}: {value}\n")
        
        return yaml_path

    def _save_best_model(self):
        """Save the best model from training"""
        runs_dir = self.database_dir / 'runs/detect'
        train_dirs = [d for d in os.listdir(runs_dir) if d.startswith('train')]
        latest_train = max(train_dirs, key=lambda x: int(x[5:]) if len(x) > 5 and x[5:].isdigit() else 0)
        best_model = runs_dir / latest_train / 'weights/best.pt'
        
        if best_model.exists():
            best_model.rename(self.final_model_path)
            print(f"Model saved as {self.final_model_path}")
        else:
            raise FileNotFoundError(f"Best model not found at {best_model}")

    def run_full_training(self, camera_source=0):
        """Run complete training pipeline"""
        print("Starting training pipeline...")
        self.setup_directories()
        self.capture_images(source=camera_source)
        self.process_images()
        
        verify = input("Verify labels? (y/n): ").lower().strip()
        if verify == 'y':
            self.verify_labels()
            
        print("Starting model training...")
        self.train_model()
        print("Training pipeline completed successfully")