"""
Face Recognition Module
Main recognition logic combining detection, embedding, and verification
"""

import cv2
import numpy as np
import time
from typing import Optional, Tuple, Dict

class FaceRecognizer:
    """
    Complete face recognition system
    Combines detection, embedding extraction, and Siamese verification
    """
    
    def __init__(self, face_detector, face_embedder, siamese_model, 
                 dataset_manager, threshold=0.6):
        """
        Args:
            face_detector: FaceDetector instance
            face_embedder: FaceEmbedder instance
            siamese_model: SiameseNetwork instance
            dataset_manager: DatasetManager instance
            threshold: Similarity threshold for verification
        """
        self.face_detector = face_detector
        self.face_embedder = face_embedder
        self.siamese_model = siamese_model
        self.dataset_manager = dataset_manager
        self.threshold = threshold
        
        # Get known embeddings
        self.known_embeddings = dataset_manager.get_all_embeddings()
        
        print(f"✓ Face Recognizer initialized")
        print(f"  Known people: {len(self.known_embeddings)}")
        print(f"  Threshold: {threshold}")
    
    def recognize_face(self, face_image: np.ndarray, 
                       return_details=False) -> Tuple[Optional[str], float, Optional[Dict]]:
        """
        Recognize a face in the image
        Args:
            face_image: Face image (H, W, 3)
            return_details: Whether to return detailed matching info
        Returns:
            (name, similarity, details) or (None, 0.0, None) if not recognized
        """
        # Extract embedding
        embedding = self.face_embedder.get_embedding(face_image)
        if embedding is None:
            return None, 0.0, None
        
        # Compare with all known embeddings
        best_match = None
        best_similarity = 0.0
        all_similarities = {}
        
        for name, known_embeddings in self.known_embeddings.items():
            # Compare with all embeddings of this person
            similarities = []
            for known_embedding in known_embeddings:
                sim = self.siamese_model.predict_similarity(embedding, known_embedding)
                similarities.append(sim)
            
            # Use maximum similarity for this person
            max_sim = max(similarities) if similarities else 0.0
            all_similarities[name] = max_sim
            
            if max_sim > best_similarity:
                best_similarity = max_sim
                best_match = name
        
        # Check threshold
        if best_similarity >= self.threshold:
            details = {
                'all_similarities': all_similarities,
                'embedding': embedding
            } if return_details else None
            
            return best_match, best_similarity, details
        
        return None, best_similarity, None
    
    def recognize_from_frame(self, frame: np.ndarray) -> list:
        """
        Recognize all faces in a frame
        Args:
            frame: Input frame (H, W, 3)
        Returns:
            List of dicts with recognition results
        """
        results = []
        
        # Detect faces
        faces = self.face_detector.detect(frame)
        
        for face_bbox in faces:
            # Extract face
            face = self.face_detector.extract_face(frame, face_bbox)
            if face is None:
                continue
            
            # Recognize
            name, similarity, _ = self.recognize_face(face)
            person_info = self.dataset_manager.get_person_info(name) if name else None
            
            results.append({
                'bbox': face_bbox,
                'face': face,
                'name': name,
                'similarity': similarity,
                'verified': name is not None,
                'info': person_info
            })
        
        return results
    
    def draw_results(self, frame: np.ndarray, results: list) -> np.ndarray:
        """
        Draw recognition results on frame
        Args:
            frame: Input frame
            results: Recognition results from recognize_from_frame
        Returns:
            Frame with drawn results
        """
        output = frame.copy()
        
        for result in results:
            x, y, w, h = result['bbox']
            name = result['name']
            similarity = result['similarity']
            verified = result['verified']
            
            # Choose color based on verification
            if verified:
                color = (0, 255, 0)  # Green for recognized
                label = f"{name} ({similarity:.2f})"
                details = result.get('info') or {}
                permission_str = "Yes" if details.get('has_permission') else "No"
                extra_lines = []
                if details:
                    if details.get('rank'):
                        extra_lines.append(f"Rank: {details.get('rank')}")
                    if details.get('age') is not None:
                        extra_lines.append(f"Age: {details.get('age')}")
                    extra_lines.append(f"Perm: {permission_str}")
                extra_text = " | ".join(extra_lines)
            else:
                color = (0, 0, 255)  # Red for unknown
                label = f"Unknown ({similarity:.2f})"
                extra_text = ""
            
            # Draw rectangle
            cv2.rectangle(output, (x, y), (x+w, y+h), color, 2)
            
            # Draw label background
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(output, (x, y-25), (x+label_size[0], y), color, -1)
            
            # Draw label text
            cv2.putText(output, label, (x, y-8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if extra_text:
                text_size, _ = cv2.getTextSize(extra_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                text_y = y + h + 25
                cv2.rectangle(output, (x, text_y - text_size[1] - 10),
                              (x + text_size[0] + 10, text_y), color, -1)
                cv2.putText(output, extra_text, (x + 5, text_y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return output
    
    def run_camera(self, camera_id=0, attendance_logger=None):
        """
        Run real-time recognition on camera feed
        Args:
            camera_id: Camera device ID
            attendance_logger: Optional AttendanceLogger instance
        """
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print("❌ Failed to open camera")
            return
        
        print("\n🎥 Starting camera recognition...")
        print("Press 'q' to quit")
        print("Press 's' to show statistics")
        
        # FPS calculation
        fps = 0
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break
            
            # Recognize faces
            results = self.recognize_from_frame(frame)
            
            # Log attendance
            if attendance_logger:
                for result in results:
                    if result['verified']:
                        attendance_logger.log_attendance(
                            result['name'],
                            result['similarity'],
                            result['face']
                        )
            
            # Draw results
            output = self.draw_results(frame, results)
            
            # Calculate and display FPS
            frame_count += 1
            if frame_count % 10 == 0:
                end_time = time.time()
                fps = 10 / (end_time - start_time)
                start_time = end_time
            
            # Display info
            info_text = f"FPS: {fps:.1f} | Faces: {len(results)}"
            cv2.putText(output, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Face Recognition - Press q to quit', output)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and attendance_logger:
                attendance_logger.print_today_attendance()
        
        cap.release()
        cv2.destroyAllWindows()
        print("✓ Camera stopped")
    
    def verify_person(self, frame: np.ndarray, expected_name: str) -> Tuple[bool, float]:
        """
        Verify if the person in frame matches expected_name
        Args:
            frame: Input frame
            expected_name: Expected person's name
        Returns:
            (verified, similarity)
        """
        results = self.recognize_from_frame(frame)
        
        for result in results:
            if result['verified'] and result['name'] == expected_name:
                return True, result['similarity']
        
        return False, 0.0
    
    def update_embeddings(self):
        """Reload embeddings from dataset manager"""
        self.known_embeddings = self.dataset_manager.get_all_embeddings()
        print(f"✓ Updated embeddings: {len(self.known_embeddings)} people")
    
    def set_threshold(self, threshold: float):
        """Update similarity threshold"""
        self.threshold = threshold
        print(f"✓ Threshold updated to {threshold}")
