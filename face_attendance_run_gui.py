"""
Run GUI Application
Main entry point for the GUI-based attendance system
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
from face_attendance_logger import AttendanceLogger
from face_attendance_recognizer import FaceRecognizer
from face_attendance_gui import AttendanceGUI
from face_attendance_antispoof import SilentFaceAntiSpoof

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
    
    print("\n3. Loading dataset...")
    dataset_manager = DatasetManager(
        DATASET_DIR,
        EMBEDDINGS_DIR,
        embedding_dim=face_embedder.embedding_dim
    )
    
    if len(dataset_manager.list_people()) == 0:
        print("\n⚠ Warning: No people in dataset")
        print("   You can add people using the GUI")
    else:
        print(f"   Loaded {len(dataset_manager.list_people())} people:")
        for person in dataset_manager.list_people():
            print(f"   - {person}")
    
    print("\n4. Initializing attendance logger...")
    attendance_logger = AttendanceLogger(
        attendance_file=Config.ATTENDANCE_FILE,
        log_interval=Config.LOG_INTERVAL,
        save_photos=True
    )
    
    print("\n5. Loading anti-spoofing model (SilentFace)...")
    anti_spoof = None
    if Config.ANTI_SPOOF_ENABLED:
        anti_spoof = SilentFaceAntiSpoof(
            model_dir=Config.ANTI_SPOOF_MODEL_PATH.parent,
            threshold=Config.ANTI_SPOOF_THRESHOLD,
            device=Config.ANTI_SPOOF_DEVICE,
            download_url=Config.ANTI_SPOOF_MODEL_URL,
        )
        print(f"   SilentFace ready (threshold={Config.ANTI_SPOOF_THRESHOLD}; model will download on first use)")
    else:
        print("   Anti-spoofing disabled via config")

    print("\n6. Initializing face recognizer...")
    face_recognizer = FaceRecognizer(
        face_detector=face_detector,
        face_embedder=face_embedder,
        dataset_manager=dataset_manager,
        threshold=Config.SIMILARITY_THRESHOLD,
        anti_spoof=anti_spoof,
    )
    
    print("\n7. Starting GUI...")
    print("="*60 + "\n")
    
    # Create and run GUI
    app = AttendanceGUI(
        face_recognizer=face_recognizer,
        dataset_manager=dataset_manager,
        attendance_logger=attendance_logger,
        face_detector=face_detector,
        face_embedder=face_embedder
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
