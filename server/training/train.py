import argparse
from .trainer import SecurityModelTrainer

def main():
    parser = argparse.ArgumentParser(description='Home Security System Training')
    parser.add_argument('--add-member', action='store_true', help='Add new home member')
    parser.add_argument('--camera', type=int, default=0, help='Camera source index')
    args = parser.parse_args()

    if args.add_member:
        trainer = SecurityModelTrainer()
        trainer.setup_directories()
        trainer.capture_images(source=args.camera)
        trainer.process_images()
        trainer.verify_labels()
        trainer.train_model()

if __name__ == '__main__':
    main()