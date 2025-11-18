"""
Add Person Script (CLI)
Command-line tool to add new people to the dataset
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from face_attendance_main import Config, DATASET_DIR, EMBEDDINGS_DIR
from face_attendance_detector import FaceDetector
from face_attendance_embedder import FaceEmbedder
from face_attendance_dataset import DatasetManager

def main():
    """Add a new person to the dataset"""
    print("\n" + "="*60)
    print("ADD NEW PERSON TO DATASET")
    print("="*60)
    
    # Get person name
    name = input("\nEnter person's name: ").strip()
    
    if not name:
        print("❌ Error: Name cannot be empty")
        sys.exit(1)
    
    # Get additional required fields
    rank = input("Enter rank: ").strip()
    if not rank:
        print("❌ Error: Rank is required")
        sys.exit(1)
    
    try:
        age_input = input("Enter age: ").strip()
        age = int(age_input)
        if age <= 0:
            raise ValueError
    except (ValueError, TypeError):
        print("❌ Error: Age must be a positive integer")
        sys.exit(1)
    
    perm_input = input("Has permission? (y/n): ").strip().lower()
    if perm_input not in ('y', 'n'):
        print("❌ Error: Permission must be 'y' or 'n'")
        sys.exit(1)
    has_permission = perm_input == 'y'
    
    # Initialize components
    print("\nInitializing system...")
    face_detector = FaceDetector(
        method=Config.FACE_DETECTION_METHOD,
        min_face_size=Config.MIN_FACE_SIZE,
        confidence=Config.DETECTION_CONFIDENCE
    )
    
    face_embedder = FaceEmbedder(model_name=Config.EMBEDDING_MODEL)
    
    dataset_manager = DatasetManager(
        DATASET_DIR,
        EMBEDDINGS_DIR,
        embedding_dim=face_embedder.embedding_dim
    )
    
    # Check if person already exists
    if name in dataset_manager.list_people():
        print(f"\n⚠ Warning: '{name}' already exists in the dataset")
        response = input("Do you want to add more images? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled")
            sys.exit(0)
    
    # Get number of images
    num_images = Config.IMAGES_PER_PERSON
    try:
        response = input(f"\nNumber of images to capture (default: {num_images}): ").strip()
        if response:
            num_images = int(response)
            if num_images < 3:
                print("⚠ Warning: Minimum 3 images recommended")
    except ValueError:
        print("⚠ Using default value")
    
    # Add person
    print("\n" + "="*60)
    success = dataset_manager.add_person(
        name=name,
        face_detector=face_detector,
        face_embedder=face_embedder,
        rank=rank,
        age=age,
        has_permission=has_permission,
        camera_id=Config.CAMERA_ID,
        num_images=num_images,
        delay=Config.CAPTURE_DELAY
    )
    
    if success:
        print("\n" + "="*60)
        print("SUCCESS!")
        print("="*60)
        print(f"✓ '{name}' has been added to the dataset")
        print(f"✓ Images saved in: {DATASET_DIR / name}")
        print(f"✓ Embeddings saved in: {EMBEDDINGS_DIR}")
        
        # Show dataset stats
        stats = dataset_manager.get_stats()
        print(f"\nDataset now contains {stats['num_people']} people:")
        for person_info in stats['people']:
            print(f"  - {person_info['name']}: {person_info['num_embeddings']} embeddings")
        
        print("\nNext steps:")
        print("  1. If this is your first person or you've added 2+ people,")
        print("     train the Siamese network: python face_attendance_train.py")
        print("  2. Run the application: python face_attendance_run_gui.py or")
        print("     python face_attendance_run_camera.py")
    else:
        print("\n" + "="*60)
        print("FAILED")
        print("="*60)
        print(f"✗ Failed to add '{name}'")
        print("Please try again")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
