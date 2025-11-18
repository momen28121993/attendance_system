"""
Run Camera Recognition (CLI)
Command-line interface for running real-time face recognition
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from face_attendance_main import Config, DATASET_DIR, EMBEDDINGS_DIR, MODELS_DIR
from face_attendance_detector import FaceDetector
from face_attendance_embedder import FaceEmbedder
from face_attendance_siamese import SiameseNetwork
from face_attendance_dataset import DatasetManager
from face_attendance_logger import AttendanceLogger
from face_attendance_recognizer import FaceRecognizer

def main():
    """Run camera recognition"""
    print("\n" + "="*60)
    print("FACE RECOGNITION ATTENDANCE SYSTEM - CAMERA")
    print("="*60)
    
    # Initialize components
    print("\n1. Initializing face detector...")
    face_detector = FaceDetector(
        method=Config.FACE_DETECTION_METHOD,
        min_face_size=Config.MIN_FACE_SIZE,
        confidence=Config.DETECTION_CONFIDENCE
    )
    
    print("\n2. Loading face embedding model...")
    face_embedder = FaceEmbedder(model_name=Config.EMBEDDING_MODEL)
    
    print("\n3. Loading Siamese network...")
    siamese = SiameseNetwork(embedding_dim=Config.EMBEDDING_DIM)
    
    if Config.SIAMESE_MODEL_PATH.exists():
        siamese.load(Config.SIAMESE_MODEL_PATH)
    else:
        print(f"\n⚠ Warning: No trained model found at {Config.SIAMESE_MODEL_PATH}")
        print("   Using untrained model. Please train the model first:")
        print("   Run: python face_attendance_train.py")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            sys.exit(0)
    
    print("\n4. Loading dataset...")
    dataset_manager = DatasetManager(
        DATASET_DIR,
        EMBEDDINGS_DIR,
        embedding_dim=face_embedder.embedding_dim
    )
    
    if len(dataset_manager.list_people()) == 0:
        print("\n❌ Error: No people in dataset")
        print("   Please add people first using:")
        print("   - python face_attendance_run_gui.py (GUI)")
        print("   - python face_attendance_add_person.py (CLI)")
        sys.exit(1)
    else:
        print(f"   Loaded {len(dataset_manager.list_people())} people:")
        for person in dataset_manager.list_people():
            embeddings = dataset_manager.get_person_embeddings(person)
            print(f"   - {person}: {len(embeddings)} embeddings")
    
    print("\n5. Initializing attendance logger...")
    attendance_logger = AttendanceLogger(
        attendance_file=Config.ATTENDANCE_FILE,
        log_interval=Config.LOG_INTERVAL,
        save_photos=True
    )
    
    print("\n6. Initializing face recognizer...")
    face_recognizer = FaceRecognizer(
        face_detector=face_detector,
        face_embedder=face_embedder,
        siamese_model=siamese,
        dataset_manager=dataset_manager,
        threshold=Config.SIMILARITY_THRESHOLD
    )
    
    print("\n" + "="*60)
    print("Starting camera...")
    print("="*60)
    print("\nControls:")
    print("  q - Quit")
    print("  s - Show today's attendance")
    print("\n")
    
    # Run camera
    try:
        face_recognizer.run_camera(
            camera_id=Config.CAMERA_ID,
            attendance_logger=attendance_logger
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Show final attendance
    print("\n" + "="*60)
    print("SESSION SUMMARY")
    print("="*60)
    attendance_logger.print_today_attendance()
    
    print("\n✓ Application closed")

if __name__ == "__main__":
    main()
