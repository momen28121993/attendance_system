"""
Face Embedding Module
Uses pre-trained models (FaceNet or MobileFaceNet) to extract face embeddings
"""

import numpy as np
import cv2
from typing import Optional
import tensorflow as tf
from tensorflow.keras.models import load_model

class FaceEmbedder:
    """Extract face embeddings using pre-trained models"""
    
    def __init__(self, model_name="facenet"):
        self.model_name = model_name.lower()
        self.model = None
        self.input_size = (160, 160) if model_name == "facenet" else (112, 112)
        self._load_model()
    
    def _load_model(self):
        """Load pre-trained embedding model"""
        if self.model_name == "facenet":
            self._load_facenet()
        elif self.model_name == "mobilefacenet":
            self._load_mobilefacenet()
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
    
    def _load_facenet(self):
        """Load FaceNet model using Keras-FaceNet or build simple model"""
        try:
            # Try to use keras-facenet if available
            from keras_facenet import FaceNet
            self.model = FaceNet()
            print("✓ FaceNet model loaded (keras-facenet)")
        except ImportError:
            # Fallback: Create a simple model for demonstration
            print("⚠ keras-facenet not found. Using simplified embedding model.")
            print("  Install for better accuracy: pip install keras-facenet")
            self.model = self._build_simple_facenet()
            print("✓ Simplified FaceNet model created")
    
    def _build_simple_facenet(self):
        """
        Build a simplified embedding model for demonstration
        Note: For production, use a pre-trained FaceNet model
        """
        from tensorflow.keras.applications import MobileNetV2
        from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
        from tensorflow.keras.models import Model
        
        # Use MobileNetV2 as backbone
        base_model = MobileNetV2(
            input_shape=(160, 160, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Add embedding layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation=None, name='embeddings')(x)
        
        # L2 normalize embeddings
        x = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x)
        
        model = Model(inputs=base_model.input, outputs=x)
        return model
    
    def _load_mobilefacenet(self):
        """Load MobileFaceNet model"""
        print("⚠ MobileFaceNet loading not implemented.")
        print("  Using simplified model instead.")
        self.model = self._build_simple_mobilefacenet()
        print("✓ Simplified MobileFaceNet model created")
    
    def _build_simple_mobilefacenet(self):
        """Build simplified MobileFaceNet"""
        from tensorflow.keras.applications import MobileNetV2
        from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
        from tensorflow.keras.models import Model
        
        base_model = MobileNetV2(
            input_shape=(112, 112, 3),
            include_top=False,
            weights='imagenet'
        )
        
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation=None)(x)
        x = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x)
        
        model = Model(inputs=base_model.input, outputs=x)
        return model
    
    def preprocess_face(self, face: np.ndarray) -> np.ndarray:
        """
        Preprocess face for embedding extraction
        Args:
            face: Face image (H, W, 3)
        Returns:
            Preprocessed face ready for model
        """
        # Resize to model input size
        face = cv2.resize(face, self.input_size)
        
        # Convert to RGB if needed
        if len(face.shape) == 2:
            face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
        elif face.shape[2] == 4:
            face = cv2.cvtColor(face, cv2.COLOR_BGRA2RGB)
        else:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        
        # Normalize
        face = face.astype('float32')
        
        # Standard normalization for pre-trained models
        mean = np.array([127.5, 127.5, 127.5])
        std = np.array([128.0, 128.0, 128.0])
        face = (face - mean) / std
        
        return face
    
    def get_embedding(self, face: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract embedding from face image
        Args:
            face: Face image (H, W, 3)
        Returns:
            Face embedding vector or None if error
        """
        try:
            # Preprocess
            face = self.preprocess_face(face)
            
            # Add batch dimension
            face = np.expand_dims(face, axis=0)
            
            # Get embedding
            if self.model_name == "facenet" and hasattr(self.model, 'embeddings'):
                # keras-facenet specific
                embedding = self.model.embeddings(face)
            else:
                # Standard Keras model
                embedding = self.model.predict(face, verbose=0)
            
            # Return as 1D array
            return embedding.flatten()
        
        except Exception as e:
            print(f"Error extracting embedding: {e}")
            return None
    
    def get_embeddings_batch(self, faces: list) -> np.ndarray:
        """
        Extract embeddings for multiple faces
        Args:
            faces: List of face images
        Returns:
            Array of embeddings (N, embedding_dim)
        """
        if not faces:
            return np.array([])
        
        # Preprocess all faces
        processed = [self.preprocess_face(face) for face in faces]
        batch = np.array(processed)
        
        # Get embeddings
        if self.model_name == "facenet" and hasattr(self.model, 'embeddings'):
            embeddings = self.model.embeddings(batch)
        else:
            embeddings = self.model.predict(batch, verbose=0)
        
        return embeddings
    
    @property
    def embedding_dim(self):
        """Get embedding dimension"""
        return 512 if self.model_name == "facenet" else 128
