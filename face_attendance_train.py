"""
Training Script
Train the Siamese network on the dataset
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from face_attendance_main import Config, DATASET_DIR, EMBEDDINGS_DIR, MODELS_DIR
from face_attendance_detector import FaceDetector
from face_attendance_embedder import FaceEmbedder
from face_attendance_siamese import SiameseNetwork, create_training_pairs
from face_attendance_dataset import DatasetManager

def train_siamese_model():
    """Train Siamese network on existing dataset"""
    print("\n" + "="*60)
    print("TRAINING SIAMESE NETWORK")
    print("="*60)
    
    # Initialize components
    print("\n1. Initializing components...")
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
    
    # Check if we have enough data
    embeddings = dataset_manager.get_all_embeddings()
    
    if len(embeddings) < 2:
        print("\n❌ Error: Need at least 2 people in dataset to train Siamese network")
        print("   Please add more people using the GUI or data collection script")
        return False
    
    print(f"\n✓ Dataset loaded: {len(embeddings)} people")
    for name, embs in embeddings.items():
        print(f"  - {name}: {len(embs)} embeddings")
    
    # Create training pairs
    print("\n2. Creating training pairs...")
    pairs, labels = create_training_pairs(embeddings, num_pairs_per_person=20)
    
    if len(pairs) == 0:
        print("\n❌ Failed to create training pairs")
        return False
    
    # Initialize and train Siamese network
    print("\n3. Building Siamese network...")
    siamese = SiameseNetwork(embedding_dim=Config.EMBEDDING_DIM)
    
    print("\n4. Training Siamese network...")
    print("   This may take a few minutes...\n")
    
    history = siamese.train(
        pairs=pairs,
        labels=labels,
        validation_split=0.2,
        epochs=50,
        batch_size=32
    )
    
    # Save model
    print(f"\n5. Saving model to {Config.SIAMESE_MODEL_PATH}...")
    Config.SIAMESE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    siamese.save(Config.SIAMESE_MODEL_PATH)
    
    # Print training summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Total pairs trained: {len(pairs)}")
    print(f"Final loss: {history.history['loss'][-1]:.4f}")
    print(f"Final accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final val_loss: {history.history['val_loss'][-1]:.4f}")
    print(f"Final val_accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"\n✓ Model saved to: {Config.SIAMESE_MODEL_PATH}")
    print("="*60)
    
    return True

if __name__ == "__main__":
    success = train_siamese_model()
    
    if success:
        print("\n✓ Training completed successfully!")
        print("\nNext steps:")
        print("  1. Run 'python face_attendance_run_gui.py' to start the GUI application")
        print("  2. Or run 'python face_attendance_run_camera.py' for command-line recognition")
    else:
        print("\n✗ Training failed. Please check the errors above.")
        sys.exit(1)
