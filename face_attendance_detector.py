"""
Face Detection Module
Supports MTCNN, OpenCV Haar Cascade, and MediaPipe
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional

class FaceDetector:
    """Unified interface for different face detection methods"""
    
    def __init__(self, method="mtcnn", min_face_size=40, confidence=0.9):
        self.method = method.lower()
        self.min_face_size = min_face_size
        self.confidence = confidence
        self.detector = None
        
        self._initialize_detector()
    
    def _initialize_detector(self):
        """Initialize the selected detector"""
        if self.method == "mtcnn":
            try:
                from mtcnn import MTCNN
                # mtcnn==1.0.0 does not expose min_face_size, so we filter manually
                self.detector = MTCNN()
                print(f"✓ MTCNN detector initialized")
            except ImportError:
                print("⚠ MTCNN not available. Install: pip install mtcnn tensorflow")
                self._fallback_to_opencv()
        
        elif self.method == "mediapipe":
            try:
                import mediapipe as mp
                self.mp_face_detection = mp.solutions.face_detection
                self.detector = self.mp_face_detection.FaceDetection(
                    model_selection=1,
                    min_detection_confidence=self.confidence
                )
                print(f"✓ MediaPipe detector initialized")
            except ImportError:
                print("⚠ MediaPipe not available. Install: pip install mediapipe")
                self._fallback_to_opencv()
        
        elif self.method == "opencv":
            self._initialize_opencv()
        
        else:
            print(f"⚠ Unknown method '{self.method}', using OpenCV")
            self._initialize_opencv()
    
    def _initialize_opencv(self):
        """Initialize OpenCV Haar Cascade detector"""
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.detector = cv2.CascadeClassifier(cascade_path)
        if self.detector.empty():
            raise RuntimeError("Failed to load OpenCV Haar Cascade")
        self.method = "opencv"
        print(f"✓ OpenCV detector initialized")
    
    def _fallback_to_opencv(self):
        """Fallback to OpenCV if other methods fail"""
        print("→ Falling back to OpenCV...")
        self._initialize_opencv()
    
    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in frame
        Returns: List of (x, y, w, h) bounding boxes
        """
        if self.method == "mtcnn":
            return self._detect_mtcnn(frame)
        elif self.method == "mediapipe":
            return self._detect_mediapipe(frame)
        elif self.method == "opencv":
            return self._detect_opencv(frame)
        return []
    
    def _detect_mtcnn(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using MTCNN"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = self.detector.detect_faces(rgb_frame)
        
        faces = []
        for detection in detections:
            if detection['confidence'] >= self.confidence:
                x, y, w, h = detection['box']
                # Ensure positive dimensions
                x, y = max(0, x), max(0, y)
                w, h = max(1, w), max(1, h)
                if w < self.min_face_size or h < self.min_face_size:
                    continue
                faces.append((x, y, w, h))
        
        return faces
    
    def _detect_mediapipe(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using MediaPipe"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb_frame)
        
        faces = []
        if results.detections:
            h, w = frame.shape[:2]
            for detection in results.detections:
                if detection.score[0] >= self.confidence:
                    bbox = detection.location_data.relative_bounding_box
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    
                    # Ensure positive dimensions
                    x, y = max(0, x), max(0, y)
                    width = max(1, width)
                    height = max(1, height)
                    
                    faces.append((x, y, width, height))
        
        return faces
    
    def _detect_opencv(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using OpenCV"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(self.min_face_size, self.min_face_size)
        )
        
        return [tuple(face) for face in faces]
    
    def extract_face(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], 
                    target_size: Tuple[int, int] = (160, 160)) -> Optional[np.ndarray]:
        """
        Extract and preprocess face from frame
        Args:
            frame: Input image
            bbox: (x, y, w, h) bounding box
            target_size: Output size for the face
        Returns:
            Preprocessed face image or None
        """
        x, y, w, h = bbox
        
        # Add margin (10%)
        margin = int(0.1 * max(w, h))
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(frame.shape[1], x + w + margin)
        y2 = min(frame.shape[0], y + h + margin)
        
        face = frame[y1:y2, x1:x2]
        
        if face.size == 0:
            return None
        
        # Resize to target size
        face = cv2.resize(face, target_size)
        return face
    
    def __del__(self):
        """Cleanup"""
        if self.method == "mediapipe" and self.detector:
            self.detector.close()
