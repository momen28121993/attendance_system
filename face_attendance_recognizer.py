"""
Face Recognition Module
ArcFace embedding + cosine similarity verification.
"""

import cv2
import numpy as np
import time
from typing import Optional, Tuple, Dict, List


class FaceRecognizer:
    """
    Complete face recognition system using ArcFace embeddings and cosine similarity.
    """

    def __init__(self, face_detector, face_embedder, dataset_manager, threshold=0.65):
        """
        Args:
            face_detector: FaceDetector instance
            face_embedder: FaceEmbedder instance
            dataset_manager: DatasetManager instance
            threshold: Cosine similarity threshold for verification
        """
        self.face_detector = face_detector
        self.face_embedder = face_embedder
        self.dataset_manager = dataset_manager
        self.threshold = threshold

        self.known_embeddings = self._prepare_embedding_cache()

        print("✓ Face Recognizer initialized")
        print(f"  Known people: {len(self.known_embeddings)}")
        print(f"  Threshold: {threshold}")

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        vec = np.asarray(vec, dtype="float32")
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm

    def _prepare_embedding_cache(self) -> Dict[str, List[np.ndarray]]:
        """Normalize all stored embeddings for cosine similarity."""
        cache: Dict[str, List[np.ndarray]] = {}
        raw_embeddings = self.dataset_manager.get_all_embeddings()
        for name, embeddings in raw_embeddings.items():
            normalized = [self._normalize(emb) for emb in embeddings if emb is not None]
            if normalized:
                cache[name] = normalized
        return cache

    def recognize_face(
        self, face_image: np.ndarray, return_details: bool = False
    ) -> Tuple[Optional[str], float, Optional[Dict]]:
        """
        Recognize a face in the image.
        """
        embedding = self.face_embedder.get_embedding(face_image)
        if embedding is None:
            return None, 0.0, None

        embedding = self._normalize(embedding)

        best_match = None
        best_similarity = -1.0
        all_similarities: Dict[str, float] = {}

        for name, known_embeddings in self.known_embeddings.items():
            sims = [float(np.dot(embedding, known_emb)) for known_emb in known_embeddings]
            if not sims:
                continue
            max_sim = max(sims)
            all_similarities[name] = max_sim
            if max_sim > best_similarity:
                best_similarity = max_sim
                best_match = name

        if best_similarity >= self.threshold:
            details = (
                {"all_similarities": all_similarities, "embedding": embedding}
                if return_details
                else None
            )
            return best_match, best_similarity, details

        return None, best_similarity if best_similarity > 0 else 0.0, None
    
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
            details = result.get('info') or {}
            has_perm = details.get('has_permission')
            if verified:
                label = f"{name} ({similarity:.2f})"
                if has_perm:
                    color = (0, 255, 0)  # green for permitted
                else:
                    color = (0, 0, 255)  # red for no permission
                extra_lines = []
                if details:
                    rank = details.get('rank') or "N/A"
                    position = details.get('position') or "N/A"
                    perm_str = "Yes" if has_perm else "No"
                    extra_lines = [
                        f"{name} | Rank: {rank}",
                        f"Position: {position}",
                        f"Perm: {perm_str}",
                    ]
                extra_text_lines = extra_lines
            else:
                color = (0, 255, 255)  # yellow for unknown
                label = f"Unknown ({similarity:.2f})"
                extra_text_lines = []
            
            # Draw rectangle
            cv2.rectangle(output, (x, y), (x+w, y+h), color, 2)
            
            # Draw label background
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(output, (x, y-25), (x+label_size[0], y), color, -1)
            
            # Draw label text
            cv2.putText(output, label, (x, y-8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            if extra_text_lines:
                text_y = y + h + 25
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in extra_text_lines]
                max_width = max(w for w, _ in sizes)
                line_height = max(h for _, h in sizes) + 4
                box_height = line_height * len(extra_text_lines) + 6
                cv2.rectangle(output, (x, text_y - box_height),
                              (x + max_width + 10, text_y), color, -1)
                current_y = text_y - box_height + line_height
                for line in extra_text_lines:
                    cv2.putText(output, line, (x + 5, current_y - 2),
                                font, font_scale, (0, 0, 0), thickness)
                    current_y += line_height
        
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
                            result["name"], result["similarity"], result["face"]
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
        self.known_embeddings = self._prepare_embedding_cache()
        print(f"✓ Updated embeddings: {len(self.known_embeddings)} people")
    
    def set_threshold(self, threshold: float):
        """Update similarity threshold"""
        self.threshold = threshold
        print(f"✓ Threshold updated to {threshold}")
