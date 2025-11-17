"""
Siamese Network Module
Compares face embeddings for verification
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from typing import Tuple, List
from pathlib import Path
import pickle
import zipfile
import tempfile
import os
import h5py

class SiameseNetwork:
    """
    Siamese Network for face verification
    Takes two embeddings and outputs similarity score
    """
    
    def __init__(self, embedding_dim=512):
        self.embedding_dim = embedding_dim
        self.model = None
        self.build_model()
    
    def build_model(self):
        """Build Siamese network architecture"""
        # Input layers for two embeddings
        input_1 = layers.Input(shape=(self.embedding_dim,), name='embedding_1')
        input_2 = layers.Input(shape=(self.embedding_dim,), name='embedding_2')
        
        # Compute L1 distance between embeddings
        l1_distance = layers.Lambda(
            lambda tensors: tf.abs(tensors[0] - tensors[1])
        )([input_1, input_2])
        
        # Dense layers for similarity learning (match trained model layout)
        x = layers.Dense(128, activation='relu', name='dense')(l1_distance)
        x = layers.Dropout(0.3, name='dropout')(x)
        x = layers.Dense(64, activation='relu', name='dense_1')(x)
        x = layers.Dropout(0.2, name='dropout_1')(x)
        x = layers.Dense(16, activation='relu', name='dense_2')(x)
        
        # Output: similarity score (sigmoid for 0-1 range)
        output = layers.Dense(1, activation='sigmoid', name='similarity')(x)
        
        self.model = Model(inputs=[input_1, input_2], outputs=output)
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        print("✓ Siamese network built")
        print(f"  Input: Two embeddings of dimension {self.embedding_dim}")
        print(f"  Output: Similarity score (0-1)")
    
    def train(self, pairs: List[Tuple[np.ndarray, np.ndarray]], 
              labels: np.ndarray, 
              validation_split=0.2, 
              epochs=50, 
              batch_size=32):
        """
        Train the Siamese network
        Args:
            pairs: List of (embedding1, embedding2) tuples
            labels: Array of labels (1 for same person, 0 for different)
            validation_split: Fraction for validation
            epochs: Training epochs
            batch_size: Batch size
        """
        # Convert pairs to arrays
        embeddings_1 = np.array([pair[0] for pair in pairs])
        embeddings_2 = np.array([pair[1] for pair in pairs])
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7
            )
        ]
        
        # Train
        history = self.model.fit(
            [embeddings_1, embeddings_2],
            labels,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print("✓ Training completed")
        return history
    
    def predict_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Predict similarity between two embeddings
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
        Returns:
            Similarity score (0-1)
        """
        # Ensure 2D shape
        if embedding1.ndim == 1:
            embedding1 = np.expand_dims(embedding1, axis=0)
        if embedding2.ndim == 1:
            embedding2 = np.expand_dims(embedding2, axis=0)
        
        # Predict
        similarity = self.model.predict([embedding1, embedding2], verbose=0)
        return float(similarity[0][0])
    
    def verify(self, embedding1: np.ndarray, embedding2: np.ndarray, threshold=0.6) -> bool:
        """
        Verify if two embeddings belong to same person
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            threshold: Similarity threshold
        Returns:
            True if same person, False otherwise
        """
        similarity = self.predict_similarity(embedding1, embedding2)
        return similarity >= threshold
    
    def save(self, filepath):
        """Save model to file"""
        self.model.save(filepath)
        print(f"✓ Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model from file"""
        filepath = Path(filepath)
        try:
            try:
                self.model = keras.models.load_model(filepath, safe_mode=False)
            except TypeError:
                # Older TensorFlow builds do not support safe_mode parameter
                self.model = keras.models.load_model(filepath)
            print(f"✓ Model loaded from {filepath}")
            return
        except Exception as exc:
            print(f"⚠ Direct model load failed ({exc}). Trying weights-only load...")
        
        # Fall back to loading weights (useful when Lambda layers were saved with
        # an incompatible Python version).
        weight_source = filepath
        temp_file = None
        try:
            if filepath.suffix == ".keras":
                with zipfile.ZipFile(filepath) as zf:
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".h5")
                    with temp_file:
                        temp_file.write(zf.read("model.weights.h5"))
                    weight_source = Path(temp_file.name)
            self._load_weights_from_h5(weight_source)
            print(f"✓ Loaded weights from {filepath}")
        finally:
            if temp_file is not None and Path(temp_file.name).exists():
                os.unlink(temp_file.name)

    def _load_weights_from_h5(self, h5_path: Path):
        """Manual loader for weights saved in functional .keras archives."""
        with h5py.File(h5_path, 'r') as f:
            if 'model_weights' in f:
                # Standard layout from `model.save_weights`
                self.model.load_weights(h5_path)
                return
            if 'layers' not in f:
                raise ValueError("No recognizable weights found in archive")
            
            layers_group = f['layers']
            layer_map = {layer.name: layer for layer in self.model.layers}
            name_map = {'dense_3': 'similarity'}
            
            for layer_name, group in layers_group.items():
                if 'vars' not in group:
                    continue
                vars_group = group['vars']
                weights = []
                for key in sorted(vars_group.keys(), key=lambda x: int(x)):
                    weights.append(vars_group[key][()])
                target_name = name_map.get(layer_name, layer_name)
                if weights and target_name in layer_map:
                    layer_map[target_name].set_weights(weights)
    
    def summary(self):
        """Print model summary"""
        self.model.summary()


def create_training_pairs(embeddings_dict: dict, num_pairs_per_person=10) -> Tuple[List, np.ndarray]:
    """
    Create training pairs from embeddings dictionary
    Args:
        embeddings_dict: {person_name: [embedding1, embedding2, ...]}
        num_pairs_per_person: Number of positive pairs per person
    Returns:
        pairs: List of (embedding1, embedding2) tuples
        labels: Array of labels (1 for same, 0 for different)
    """
    pairs = []
    labels = []
    
    names = list(embeddings_dict.keys())
    
    if len(names) < 2:
        print("⚠ Need at least 2 people to create training pairs")
        return pairs, np.array(labels)
    
    # Create positive pairs (same person)
    for name in names:
        embeddings = embeddings_dict[name]
        if len(embeddings) < 2:
            continue
        
        for _ in range(num_pairs_per_person):
            idx1, idx2 = np.random.choice(len(embeddings), 2, replace=False)
            pairs.append((embeddings[idx1], embeddings[idx2]))
            labels.append(1)
    
    # Create negative pairs (different persons)
    num_positive = len([l for l in labels if l == 1])
    num_negative = num_positive  # Balance dataset
    
    for _ in range(num_negative):
        name1, name2 = np.random.choice(names, 2, replace=False)
        emb1 = np.random.choice(embeddings_dict[name1])
        emb2 = np.random.choice(embeddings_dict[name2])
        pairs.append((emb1, emb2))
        labels.append(0)
    
    # Shuffle
    indices = np.random.permutation(len(pairs))
    pairs = [pairs[i] for i in indices]
    labels = np.array([labels[i] for i in indices])
    
    print(f"✓ Created {len(pairs)} training pairs")
    print(f"  Positive pairs: {np.sum(labels)}")
    print(f"  Negative pairs: {len(labels) - np.sum(labels)}")
    
    return pairs, labels
