"""
Dataset Management Module
Handles face dataset creation, storage, and embedding management
"""

import os
import cv2
import numpy as np
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import shutil

class DatasetManager:
    """Manage face dataset and embeddings"""
    
    def __init__(self, dataset_dir: Path, embeddings_dir: Path):
        self.dataset_dir = Path(dataset_dir)
        self.embeddings_dir = Path(embeddings_dir)
        self.embeddings_file = self.embeddings_dir / "embeddings.pkl"
        self.metadata_file = self.embeddings_dir / "metadata.json"
        
        # Ensure directories exist
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing embeddings
        self.embeddings_cache = self.load_embeddings()
        self.metadata = self.load_metadata()
    
    def add_person(self, name: str, face_detector, face_embedder,
                   rank: str, age: int, has_permission: bool,
                   camera_id=0, num_images=10, delay=0.5) -> bool:
        """
        Add a new person to the dataset by capturing faces from camera
        Args:
            name: Person's name
            face_detector: FaceDetector instance
            face_embedder: FaceEmbedder instance
            rank: Person's rank/role
            age: Person's age
            has_permission: Whether person has access permission
            camera_id: Camera device ID
            num_images: Number of images to capture
            delay: Delay between captures (seconds)
        Returns:
            Success status
        """
        # Create person directory
        person_dir = self.dataset_dir / name
        person_dir.mkdir(exist_ok=True)
        
        # Initialize camera
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print("❌ Failed to open camera")
            return False
        
        print(f"\n📸 Capturing {num_images} images for '{name}'")
        print("Instructions:")
        print("  - Look at the camera")
        print("  - Move your head slightly for different angles")
        print("  - Press 'q' to quit early")
        print("\nStarting in 3 seconds...")
        time.sleep(3)
        
        captured = 0
        embeddings = []
        last_capture_time = 0
        
        while captured < num_images:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break
            
            # Detect faces
            faces = face_detector.detect(frame)
            
            # Display
            display_frame = frame.copy()
            
            if len(faces) > 0:
                # Use largest face
                face_bbox = max(faces, key=lambda f: f[2] * f[3])
                x, y, w, h = face_bbox
                
                # Draw rectangle
                color = (0, 255, 0) if time.time() - last_capture_time > delay else (0, 165, 255)
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                
                # Capture if delay passed
                current_time = time.time()
                if current_time - last_capture_time > delay:
                    # Extract face
                    face = face_detector.extract_face(frame, face_bbox)
                    if face is not None:
                        # Save image
                        img_path = person_dir / f"{name}_{captured+1:03d}.jpg"
                        cv2.imwrite(str(img_path), face)
                        
                        # Get embedding
                        embedding = face_embedder.get_embedding(face)
                        if embedding is not None:
                            embeddings.append(embedding)
                            captured += 1
                            last_capture_time = current_time
                            print(f"  ✓ Captured {captured}/{num_images}")
            
            # Display progress
            progress_text = f"Captured: {captured}/{num_images}"
            cv2.putText(display_frame, progress_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Add Person - Press q to quit', display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n⚠ Capture interrupted by user")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if captured < 3:
            print(f"❌ Insufficient images captured ({captured}). Need at least 3.")
            return False
        
        # Update embeddings cache
        self.embeddings_cache[name] = embeddings
        self.save_embeddings()
        
        # Update metadata
        self.metadata[name] = {
            'rank': rank,
            'age': age,
            'has_permission': has_permission,
            'num_images': captured,
            'num_embeddings': len(embeddings),
            'added_timestamp': time.time(),
            'directory': str(person_dir)
        }
        self.save_metadata()
        
        print(f"✓ Successfully added '{name}' with {captured} images")
        return True
    
    def load_embeddings(self) -> Dict[str, List[np.ndarray]]:
        """Load embeddings from disk"""
        if self.embeddings_file.exists():
            try:
                with open(self.embeddings_file, 'rb') as f:
                    embeddings = pickle.load(f)
                print(f"✓ Loaded embeddings for {len(embeddings)} people")
                return embeddings
            except Exception as e:
                print(f"⚠ Error loading embeddings: {e}")
        return {}
    
    def save_embeddings(self):
        """Save embeddings to disk"""
        try:
            with open(self.embeddings_file, 'wb') as f:
                pickle.dump(self.embeddings_cache, f)
            print(f"✓ Saved embeddings for {len(self.embeddings_cache)} people")
        except Exception as e:
            print(f"❌ Error saving embeddings: {e}")
    
    def load_metadata(self) -> Dict:
        """Load metadata from disk"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠ Error loading metadata: {e}")
        return {}
    
    def save_metadata(self):
        """Save metadata to disk"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            print(f"❌ Error saving metadata: {e}")
    
    def get_all_embeddings(self) -> Dict[str, List[np.ndarray]]:
        """Get all embeddings"""
        return self.embeddings_cache
    
    def get_person_embeddings(self, name: str) -> List[np.ndarray]:
        """Get embeddings for a specific person"""
        return self.embeddings_cache.get(name, [])
    
    def rebuild_embeddings(self, face_detector, face_embedder):
        """
        Rebuild embeddings from images in dataset directory
        Useful when changing embedding model
        """
        print("\n🔄 Rebuilding embeddings from dataset...")
        new_embeddings = {}
        
        for person_dir in self.dataset_dir.iterdir():
            if not person_dir.is_dir():
                continue
            
            name = person_dir.name
            embeddings = []
            
            # Process all images
            image_files = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png"))
            print(f"  Processing {name}: {len(image_files)} images...")
            
            for img_path in image_files:
                # Load image
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                # Detect face
                faces = face_detector.detect(img)
                if len(faces) == 0:
                    continue
                
                # Extract face
                face = face_detector.extract_face(img, faces[0])
                if face is None:
                    continue
                
                # Get embedding
                embedding = face_embedder.get_embedding(face)
                if embedding is not None:
                    embeddings.append(embedding)
            
            if embeddings:
                new_embeddings[name] = embeddings
                print(f"    ✓ Generated {len(embeddings)} embeddings")
        
        # Update cache
        self.embeddings_cache = new_embeddings
        self.save_embeddings()
        
        # Update metadata
        for name, embeddings in new_embeddings.items():
            existing_meta = self.metadata.get(name, {})
            preserved_meta = {
                key: existing_meta.get(key)
                for key in ['rank', 'age', 'has_permission', 'added_timestamp', 'directory', 'num_images']
                if key in existing_meta
            }
            person_dir = self.dataset_dir / name
            preserved_meta.setdefault('directory', str(person_dir))
            preserved_meta['num_embeddings'] = len(embeddings)
            preserved_meta['rebuilt_timestamp'] = time.time()
            self.metadata[name] = preserved_meta
        self.save_metadata()
        
        print(f"✓ Embeddings rebuilt for {len(new_embeddings)} people")
    
    def list_people(self) -> List[str]:
        """List all people in dataset"""
        return sorted(self.embeddings_cache.keys())
    
    def remove_person(self, name: str, delete_files: bool = True) -> bool:
        """Remove a person from dataset and optionally delete their images"""
        removed = False
        
        if name in self.embeddings_cache:
            del self.embeddings_cache[name]
            self.save_embeddings()
            removed = True
        
        if name in self.metadata:
            del self.metadata[name]
            self.save_metadata()
            removed = True
        
        if delete_files:
            person_dir = self.dataset_dir / name
            if person_dir.exists():
                shutil.rmtree(person_dir)
                removed = True
        
        if removed:
            print(f"✓ Removed '{name}' from dataset")
        else:
            print(f"⚠ Person '{name}' not found in dataset")
        return removed
    
    def get_stats(self) -> Dict:
        """Get dataset statistics"""
        stats = {
            'num_people': len(self.embeddings_cache),
            'people': []
        }
        
        for name, embeddings in self.embeddings_cache.items():
            person_stats = {
                'name': name,
                'num_embeddings': len(embeddings),
                'metadata': self.metadata.get(name, {})
            }
            stats['people'].append(person_stats)
        
        return stats

    def get_person_info(self, name: str) -> Optional[Dict]:
        """Get detailed info for a single person"""
        embeddings = self.embeddings_cache.get(name)
        if embeddings is None:
            return None
        
        person_dir = self.dataset_dir / name
        image_count = 0
        if person_dir.exists():
            image_count = len(list(person_dir.glob("*.jpg"))) + len(list(person_dir.glob("*.png")))
        
        meta = self.metadata.get(name, {})
        info = {
            'name': name,
            'rank': meta.get('rank', ''),
            'age': meta.get('age'),
            'has_permission': meta.get('has_permission'),
            'num_embeddings': len(embeddings),
            'num_images': meta.get('num_images', image_count),
            'directory': meta.get('directory', str(person_dir)),
            'metadata': meta
        }
        return info
