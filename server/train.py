import os, random, argparse, requests
import cv2, face_recognition
from ultralytics import YOLO


def capture_images(output_root = 'database', split_ratio=0.8, source=0):
    train_dir = os.path.join(output_root, 'train')
    val_dir = os.path.join(output_root, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    cap = cv2.VideoCapture(source)
    image_count = 0
    train_count = 0
    val_count = 0

    print("Press 'c' to capture an image")
    print("Press 'q' to quit")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            print(f"Error: Could not access camera source {source}")
            break
        
        cv2.imshow("Preview", frame)
        key = cv2.waitKey(1)
        
        if key == ord('c'):
            if random.random() < split_ratio:
                dest_dir = train_dir
                train_count += 1
            else:
                dest_dir = val_dir
                val_count += 1
            
            img_path = os.path.join(dest_dir, f"image_{image_count}.jpg")
            cv2.imwrite(img_path, frame)
            print(f"Captured: {img_path} (Train: {train_count}, Val: {val_count})")
            image_count += 1
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def convert_to_yolo_bbox(face_location, img_width, img_height):
    top, right, bottom, left = face_location
    x_center = (left + right) / 2.0 / img_width
    y_center = (top + bottom) / 2.0 / img_height
    width = (right - left) / img_width
    height = (bottom - top) / img_height
    return (x_center, y_center, width, height)


def process_images(images_dir='database/train', labels_dir='database/train/labels', annotated_dir='database/train/annotated', class_id=0):
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(annotated_dir, exist_ok=True)

    for img_name in os.listdir(images_dir):
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(images_dir, img_name)
        image = cv2.imread(img_path)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        face_locations = face_recognition.face_locations(rgb_image)
        if not face_locations:
            continue

        height, width = image.shape[:2]
        bbox = convert_to_yolo_bbox(face_locations[0], width, height)
        
        label_path = os.path.join(labels_dir, f"{os.path.splitext(img_name)[0]}.txt")
        with open(label_path, 'w') as f:
            f.write(f"{class_id} {' '.join(map(str, bbox))}\n")
        
        top, right, bottom, left = face_locations[0]
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.imwrite(os.path.join(annotated_dir, img_name), image)


def verify_labels(images_root='database', mode='both'):
    modes = []
    if mode in ['train', 'both']:
        modes.append('train')
    if mode in ['val', 'both']:
        modes.append('val')
    
    for dataset_mode in modes:
        images_dir = os.path.join(images_root, dataset_mode, 'images')  
        labels_dir = os.path.join(images_root, dataset_mode, 'labels') 

        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            print(f"Skipping {dataset_mode}: directory not found")
            continue

        for img_name in os.listdir(images_dir):
            if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(images_dir, img_name)
            label_path = os.path.join(labels_dir, f"{os.path.splitext(img_name)[0]}.txt")
            
            if not os.path.exists(label_path):
                print(f"Warning: No label found for {img_name}")
                continue

            image = cv2.imread(img_path)
            if image is None:
                print(f"Warning: Could not read {img_name}")
                continue

            height, width = image.shape[:2]
            
            with open(label_path, 'r') as f:
                for line in f:
                    class_id, xc, yc, w, h = map(float, line.strip().split())
                    x = int((xc - w/2) * width)
                    y = int((yc - h/2) * height)
                    x2 = int(x + w * width)
                    y2 = int(y + h * height)
                    cv2.rectangle(image, (x, y), (x2, y2), (0, 255, 0), 2)

            cv2.imshow(f'Verification ({dataset_mode})', image)
            key = cv2.waitKey(500)
            if key == ord('q'): 
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()


def download_model(url, save_path):
    print(f"Downloading YOLO model to {save_path}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    block_size = 8192
    
    with open(save_path, 'wb') as f:
        for data in response.iter_content(block_size):
            downloaded += len(data)
            f.write(data)
            done = int(50 * downloaded / total_size)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f"\rProgress: [{'=' * done}{' ' * (50-done)}] {mb_downloaded:.1f}/{mb_total:.1f} MB", end='')
    print("\nDownload complete!")


def reorganize_dataset():
    base_dir = 'database'
    for split in ['train', 'val']:
        images_dir = os.path.join(base_dir, split, 'images')
        labels_dir = os.path.join(base_dir, split, 'labels')
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        source_dir = os.path.join(base_dir, split)
        for file in os.listdir(source_dir):
            if file.endswith(('.jpg', '.jpeg', '.png')):
                old_path = os.path.join(source_dir, file)
                new_path = os.path.join(images_dir, file)
                os.rename(old_path, new_path)

        old_labels = os.path.join(base_dir, split, 'labels')
        if os.path.exists(old_labels):
            for file in os.listdir(old_labels):
                if file.endswith('.txt'):
                    old_path = os.path.join(old_labels, file)
                    new_path = os.path.join(labels_dir, file)
                    os.rename(old_path, new_path)


def add_new_member():
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)
    os.makedirs('database', exist_ok=True)
    
    data_yaml = {
        'path': project_root,
        'train': os.path.join('database', 'train', 'images'),
        'val': os.path.join('database', 'val', 'images'),
        'nc': 1,
        'names': ['home']
    }

    yaml_path = os.path.join(project_root, 'database', 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        for key, value in data_yaml.items():
            f.write(f'{key}: {value}\n')

    capture_images()

    for split in ['train', 'val']:
        process_images(
            images_dir=f'database/{split}',
            labels_dir=f'database/{split}/labels',
            annotated_dir=f'database/{split}/annotated'
        )

    reorganize_dataset()

    verify = input("Do you want to verify the labels? (y/n): ").lower().strip()
    if verify == 'y':
        verify_labels(images_root='database')

    model_path = os.path.join(project_root, 'database', 'yolo11n.pt')
    if not os.path.exists(model_path):
        url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
        try:
            download_model(url, model_path)
        except Exception as e:
            raise FileNotFoundError(f"Failed to download model: {e}")
    
    model = YOLO(model_path)
    model.train(
        data=yaml_path,
        epochs=50,
        imgsz=640,
        project=os.path.join(project_root, 'runs'),
        name='detect/train'
    )

    runs_dir = os.path.join(project_root, 'runs', 'detect')
    train_dirs = [d for d in os.listdir(runs_dir) if d.startswith('train')]
    latest_train = max(train_dirs, key=lambda x: int(x[5:]) if len(x) > 5 and x[5:].isdigit() else 0)
    best_model = os.path.join(runs_dir, latest_train, 'weights', 'best.pt')
    
    if not os.path.exists(best_model):
        raise FileNotFoundError(f"Best model not found at {best_model}")
    else:
        print(f"Best model found at {best_model}")
        
    final_model = os.path.join(project_root, 'database', 'home.pt')
    os.replace(best_model, final_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Server for Home Security System')
    parser.add_argument('--add-member', action='store_true', help='Add a new home member to the model')
    args = parser.parse_args()

    if args.add_member:
        add_new_member()
