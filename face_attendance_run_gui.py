"""
Run GUI Application
Main entry point for the GUI-based attendance system
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
from face_attendance_gui import AttendanceGUI

def main():
    """Run the GUI application"""
    print("\n" + "="*60)
    print("FACE RECOGNITION ATTENDANCE SYSTEM - GUI")
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
    dataset_manager = DatasetManager(DATASET_DIR, EMBEDDINGS_DIR)
    
    if len(dataset_manager.list_people()) == 0:
        print("\n⚠ Warning: No people in dataset")
        print("   You can add people using the GUI")
    else:
        print(f"   Loaded {len(dataset_manager.list_people())} people:")
        for person in dataset_manager.list_people():
            print(f"   - {person}")
    
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
    
    print("\n7. Starting GUI...")
    print("="*60 + "\n")
    
    # Create and run GUI
    app = AttendanceGUI(
        face_recognizer=face_recognizer,
        dataset_manager=dataset_manager,
        attendance_logger=attendance_logger,
        face_detector=face_detector,
        face_embedder=face_embedder,
        siamese_model=siamese
    )
    
    try:
        app.run()
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✓ Application closed")

if __name__ == "__main__":
    main()
