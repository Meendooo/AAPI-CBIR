"""
CNN Feature Extractor using pre-trained deep learning models

This extractor uses transfer learning with models pre-trained on ImageNet
to extract high-level semantic features from images.
"""

import cv2
import numpy as np
from typing import Union
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image


class CNNFeaturesExtractor:
    """
    Extracts deep learning features using a pre-trained CNN (ResNet50).
    
    Uses the output of the global average pooling layer as features,
    which provides a compact 2048-dimensional representation.
    """
    
    def __init__(self, model_name: str = 'resnet50', pooling: str = 'avg'):
        """
        Initialize the CNN extractor.
        
        Args:
            model_name: Name of the pre-trained model ('resnet50')
            pooling: Type of pooling ('avg' or 'max')
        """
        self.model_name = model_name
        self.pooling = pooling
        
        # Load pre-trained model without top classification layer
        print(f"Loading {model_name} model...")
        if model_name == 'resnet50':
            self.model = ResNet50(
                weights='imagenet',
                include_top=False,
                pooling=pooling
            )
            self.feature_dim = 2048
            self.target_size = (224, 224)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        print(f"Model loaded. Feature dimension: {self.feature_dim}")
    
    def extract(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Extract CNN features from an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Normalized feature vector
        """
        # Load and preprocess image
        img = keras_image.load_img(
            str(image_path), 
            target_size=self.target_size
        )
        img_array = keras_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Extract features
        features = self.model.predict(img_array, verbose=0)
        features = features.flatten()
        
        # Normalize the feature vector (L2 normalization)
        features = self._normalize(features)
        
        return features
    
    def extract_batch(self, image_paths: list, batch_size: int = 32) -> np.ndarray:
        """
        Extract features from multiple images in batches.
        
        Args:
            image_paths: List of paths to image files
            batch_size: Number of images to process at once
            
        Returns:
            Feature matrix of shape (n_images, feature_dim)
        """
        features_list = []
        
        # Process in batches for efficiency
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            
            for img_path in batch_paths:
                try:
                    img = keras_image.load_img(
                        str(img_path), 
                        target_size=self.target_size
                    )
                    img_array = keras_image.img_to_array(img)
                    batch_images.append(img_array)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    # Add zero image for failed loads
                    batch_images.append(np.zeros((*self.target_size, 3)))
            
            # Preprocess batch
            batch_array = np.array(batch_images)
            batch_array = preprocess_input(batch_array)
            
            # Extract features for batch
            batch_features = self.model.predict(batch_array, verbose=0)
            
            # Normalize each feature vector
            for features in batch_features:
                features = features.flatten()
                features = self._normalize(features)
                features_list.append(features)
        
        return np.array(features_list, dtype=np.float32)
    
    @staticmethod
    def _normalize(features: np.ndarray) -> np.ndarray:
        """
        L2 normalize the feature vector.
        
        Args:
            features: Feature vector
            
        Returns:
            Normalized feature vector
        """
        norm = np.linalg.norm(features)
        if norm > 0:
            return features / norm
        return features
    
    def get_name(self) -> str:
        """Return the name of this extractor."""
        return "cnn_features"
    
    def get_feature_dim(self) -> int:
        """Return the dimensionality of the feature vector."""
        return self.feature_dim
