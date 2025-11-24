"""
Dataset Management Module
Handles face dataset creation, storage, and embedding management
"""

import cv2
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import shutil
import pickle
import threading

try:
    import faiss  # type: ignore
except ImportError as exc:
    raise ImportError(
        "FAISS is required for vector storage. Install it with 'pip install faiss-cpu'."
    ) from exc

class DatasetManager:
    """Manage face dataset and embeddings"""
    
    def __init__(self, dataset_dir: Path, embeddings_dir: Path, embedding_dim: int = 512):
        self.dataset_dir = Path(dataset_dir)
        self.embeddings_dir = Path(embeddings_dir)
        self.embedding_dim = embedding_dim
        self.index_file = self.embeddings_dir / "embeddings.faiss"
        self.labels_file = self.embeddings_dir / "embeddings_labels.json"
        self.metadata_file = self.embeddings_dir / "metadata.json"
        self.legacy_embeddings_file = self.embeddings_dir / "embeddings.pkl"
        
        # Ensure directories exist
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        # Synchronization for background operations
        self._lock = threading.RLock()

        # Load FAISS index/labels before caching embeddings
        self.index = self._load_faiss_index()
        self.vector_labels = self._load_vector_labels()
        
        # Load existing embeddings from FAISS
        self.embeddings_cache = self.load_embeddings()
        self.metadata = self.load_metadata()
    
    def add_person(self, name: str, face_detector, face_embedder,
                   rank: str, position: str, has_permission: bool,
                   camera_id=0, num_images=10, delay=0.5) -> bool:
        """
        Add a new person to the dataset by capturing faces from camera
        Args:
            name: Person's name
            face_detector: FaceDetector instance
            face_embedder: FaceEmbedder instance
            rank: Person's rank/role
            position: Person's position/role (text)
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
        images_saved = 0
        base_face_paths: List[Path] = []
        
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
                    face = face_detector.extract_face(
                        frame, face_bbox, target_size=face_embedder.input_size
                    )
                    if face is not None:
                        img_path = person_dir / f"{name}_{captured+1:03d}.jpg"
                        cv2.imwrite(str(img_path), face)
                        base_face_paths.append(img_path)
                        images_saved += 1
                        
                        embedding = face_embedder.get_embedding(face)
                        if embedding is not None:
                            embeddings.append(self._flatten_embedding(embedding))
                        captured += 1
                        last_capture_time = current_time
                        print(f"  ✓ Captured {captured}/{num_images}")
            else:
                cv2.putText(
                    display_frame,
                    "No face detected",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )
            
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
        
        with self._lock:
            existing = self.embeddings_cache.get(name, [])
            existing.extend(embeddings)
            self.embeddings_cache[name] = existing
            self.metadata[name] = {
                'rank': rank,
                'position': position,
                'has_permission': has_permission,
                'num_images': images_saved,
                'num_embeddings': len(existing),
                'added_timestamp': time.time(),
                'directory': str(person_dir)
            }
        self.save_embeddings()
        self.save_metadata()
        
        if base_face_paths:
            threading.Thread(
                target=self._augment_and_embed,
                args=(name, base_face_paths, face_embedder),
                daemon=True
            ).start()
        
        print(f"✓ Successfully added '{name}' with {captured} images")
        return True

    def _augment_face_samples(self, face: np.ndarray) -> List[np.ndarray]:
        """
        Generate simple augmentations to improve dataset variety.
        Includes original, horizontal flip, brightness shifts, and slight blur/noise.
        """
        samples = []
        flipped = cv2.flip(face, 1)
        if flipped is not None:
            samples.append(flipped)
        brighter = cv2.convertScaleAbs(face, alpha=1.15, beta=10)
        darker = cv2.convertScaleAbs(face, alpha=0.9, beta=-10)
        samples.extend([brighter, darker])
        blurred = cv2.GaussianBlur(face, (3, 3), 0)
        noise = np.clip(face.astype(np.int16) + np.random.normal(0, 6, face.shape), 0, 255).astype(np.uint8)
        samples.extend([blurred, noise])
        return samples

    def _augment_and_embed(self, name: str, image_paths: List[Path], face_embedder) -> None:
        augmented_embeddings = []
        images_added = 0
        for img_path in image_paths:
            face = cv2.imread(str(img_path))
            if face is None:
                continue
            for idx, aug_face in enumerate(self._augment_face_samples(face), start=1):
                aug_path = img_path.with_name(f"{img_path.stem}_aug{idx}.jpg")
                cv2.imwrite(str(aug_path), aug_face)
                embedding = face_embedder.get_embedding(aug_face)
                if embedding is not None:
                    augmented_embeddings.append(self._flatten_embedding(embedding))
                    images_added += 1
        if augmented_embeddings:
            with self._lock:
                existing = self.embeddings_cache.get(name, [])
                existing.extend(augmented_embeddings)
                self.embeddings_cache[name] = existing
                meta = self.metadata.get(name, {})
                meta['num_images'] = meta.get('num_images', 0) + images_added
                meta['num_embeddings'] = len(existing)
                self.metadata[name] = meta
            self.save_embeddings()
            self.save_metadata()

    @staticmethod
    def _flatten_embedding(embedding: np.ndarray) -> np.ndarray:
        arr = np.asarray(embedding, dtype='float32')
        return arr.reshape(-1)
    
    def _create_index(self):
        """Create an empty FAISS index"""
        return faiss.IndexFlatIP(self.embedding_dim)
    
    def _load_faiss_index(self):
        """Load FAISS index from disk or create a new one"""
        if self.index_file.exists():
            try:
                index = faiss.read_index(str(self.index_file))
                print(f"✓ Loaded FAISS index with {index.ntotal} embeddings")
                return index
            except Exception as e:
                print(f"⚠ Error loading FAISS index: {e}")
        return self._create_index()
    
    def _load_vector_labels(self) -> List[str]:
        """Load stored labels for each embedding in the FAISS index"""
        if self.labels_file.exists():
            try:
                with open(self.labels_file, 'r') as f:
                    labels = json.load(f)
                return labels
            except Exception as e:
                print(f"⚠ Error loading embedding labels: {e}")
        return []
    
    def _save_faiss_index(self):
        """Persist FAISS index and labels"""
        try:
            faiss.write_index(self.index, str(self.index_file))
            with open(self.labels_file, 'w') as f:
                json.dump(self.vector_labels, f, indent=2)
        except Exception as e:
            print(f"❌ Error saving FAISS index: {e}")
    
    def _load_legacy_embeddings(self) -> Dict[str, List[np.ndarray]]:
        """Load embeddings from the previous pickle format for migration"""
        if self.legacy_embeddings_file.exists():
            try:
                with open(self.legacy_embeddings_file, 'rb') as f:
                    data = pickle.load(f)
                print("⚠ Legacy embeddings pickle detected. Migrating to FAISS...")
                return data
            except Exception as e:
                print(f"⚠ Failed to load legacy embeddings: {e}")
        return {}
    
    def load_embeddings(self) -> Dict[str, List[np.ndarray]]:
        """Load embeddings from FAISS index"""
        embeddings: Dict[str, List[np.ndarray]] = {}
        if self.index is None or self.index.ntotal == 0:
            legacy_embeddings = self._load_legacy_embeddings()
            if legacy_embeddings:
                self.embeddings_cache = legacy_embeddings
                self.save_embeddings()
                return legacy_embeddings
            return embeddings
        
        if len(self.vector_labels) != self.index.ntotal:
            print("⚠ Label count does not match FAISS index size. Rebuilding index metadata.")
            self.vector_labels = self.vector_labels[:self.index.ntotal]
        
        for idx in range(min(len(self.vector_labels), self.index.ntotal)):
            name = self.vector_labels[idx]
            try:
                vector = self.index.reconstruct(idx)
            except Exception as e:
                print(f"⚠ Failed to reconstruct embedding {idx}: {e}")
                continue
            embeddings.setdefault(name, []).append(vector)
        
        if embeddings:
            print(f"✓ Loaded embeddings for {len(embeddings)} people from FAISS")
        return embeddings
    
    def save_embeddings(self):
        """Save embeddings to FAISS index"""
        with self._lock:
            vectors = []
            labels = []
            
            for name, person_embeddings in self.embeddings_cache.items():
                for embedding in person_embeddings:
                    if embedding is None:
                        continue
                    vectors.append(np.asarray(embedding, dtype='float32'))
                    labels.append(name)
            
            self.index = self._create_index()
            if vectors:
                matrix = np.vstack(vectors).astype('float32')
                self.index.add(matrix)
            self.vector_labels = labels
            self._save_faiss_index()
            print(f"✓ Saved {len(labels)} embeddings across {len(self.embeddings_cache)} people to FAISS")
    
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
        with self._lock:
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
                for key in ['rank', 'position', 'has_permission', 'added_timestamp', 'directory', 'num_images']
                if key in existing_meta
            }
            if 'position' not in preserved_meta and 'age' in existing_meta:
                preserved_meta['position'] = existing_meta.get('age')
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
        # Support legacy 'age' field if present
        position = meta.get('position') or meta.get('age')
        info = {
            'name': name,
            'rank': meta.get('rank', ''),
            'position': position,
            'has_permission': meta.get('has_permission'),
            'num_embeddings': len(embeddings),
            'num_images': meta.get('num_images', image_count),
            'directory': meta.get('directory', str(person_dir)),
            'metadata': meta
        }
        return info
