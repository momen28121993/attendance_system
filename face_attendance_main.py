"""
Face Recognition Attendance System
Main entry point and configuration
"""

import os
import sys
from pathlib import Path

# Project structure
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = DATA_DIR / "faces"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
DEFAULT_MODELS_DIR = DATA_DIR / "models"
EXTERNAL_MODELS_DIR = PROJECT_ROOT / "model"
MODELS_DIR = EXTERNAL_MODELS_DIR if EXTERNAL_MODELS_DIR.exists() else DEFAULT_MODELS_DIR
LOGS_DIR = DATA_DIR / "logs"
ATTENDANCE_DIR = DATA_DIR / "attendance"

# Create directories
for directory in [DATA_DIR, DATASET_DIR, EMBEDDINGS_DIR, MODELS_DIR, LOGS_DIR, ATTENDANCE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def resolve_siamese_model_path():
    """
    Prefer an existing trained model from the shared 'model' folder, falling
    back to the default h5 path inside the data directory.
    """
    keras_path = MODELS_DIR / "siamese_model.keras"
    h5_path = MODELS_DIR / "siamese_model.h5"
    for candidate in (keras_path, h5_path):
        if candidate.exists():
            return candidate
    return h5_path

# Configuration
class Config:
    # Face detection
    FACE_DETECTION_METHOD = "mtcnn"  # Options: mtcnn, opencv, mediapipe
    MIN_FACE_SIZE = 40
    DETECTION_CONFIDENCE = 0.9
    
    # Embedding model
    EMBEDDING_MODEL = "facenet"  # facenet or mobilefacenet
    EMBEDDING_DIM = 512 if EMBEDDING_MODEL == "facenet" else 128
    
    # Siamese network
    SIAMESE_MODEL_PATH = resolve_siamese_model_path()
    SIMILARITY_THRESHOLD = 0.6  # Adjust based on validation
    
    # Data collection
    IMAGES_PER_PERSON = 10
    CAPTURE_DELAY = 0.5  # seconds between captures
    
    # Attendance
    ATTENDANCE_FILE = ATTENDANCE_DIR / "attendance.csv"
    LOG_INTERVAL = 300  # seconds (5 min) - prevent duplicate logs
    
    # Camera
    CAMERA_ID = 0
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480

if __name__ == "__main__":
    print("Face Recognition Attendance System")
    print("=" * 50)
    print("\nProject Structure:")
    print(f"Root: {PROJECT_ROOT}")
    print(f"Dataset: {DATASET_DIR}")
    print(f"Embeddings: {EMBEDDINGS_DIR}")
    print(f"Models: {MODELS_DIR}")
    print(f"Logs: {LOGS_DIR}")
    print(f"Attendance: {ATTENDANCE_DIR}")
    print("\nConfiguration loaded successfully!")
