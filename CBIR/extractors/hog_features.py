"""
HOG (Histogram of Oriented Gradients) Feature Extractor

HOG is a feature descriptor that captures the structure and shape of objects
by analyzing the distribution of gradient orientations.
"""

import cv2
import numpy as np
from typing import Union
from pathlib import Path
from skimage.feature import hog


class HOGFeaturesExtractor:
    """
    Extracts HOG (Histogram of Oriented Gradients) features from images.
    
    HOG captures edge and gradient structure, making it effective for
    shape and object recognition.
    """
    
    def __init__(self, 
                 orientations: int = 9,
                 pixels_per_cell: tuple = (8, 8),
                 cells_per_block: tuple = (2, 2),
                 resize_shape: tuple = (128, 128)):
        """
        Initialize the HOG extractor.
        
        Args:
            orientations: Number of orientation bins (default: 9)
            pixels_per_cell: Size of a cell in pixels (default: 8x8)
            cells_per_block: Number of cells in each block (default: 2x2)
            resize_shape: Target size for images (default: 128x128)
        """
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.resize_shape = resize_shape
        
        # Calculate feature dimension
        # For a 128x128 image with 8x8 cells and 2x2 blocks:
        # cells = (128/8, 128/8) = (16, 16)
        # blocks = (16-2+1, 16-2+1) = (15, 15)
        # features per block = 2*2*9 = 36
        # total = 15*15*36 = 8100
        cells_x = resize_shape[0] // pixels_per_cell[0]
        cells_y = resize_shape[1] // pixels_per_cell[1]
        blocks_x = cells_x - cells_per_block[0] + 1
        blocks_y = cells_y - cells_per_block[1] + 1
        features_per_block = cells_per_block[0] * cells_per_block[1] * orientations
        self.feature_dim = blocks_x * blocks_y * features_per_block
        
    def extract(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Extract HOG features from an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Normalized feature vector
        """
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Convert to grayscale (HOG works on grayscale)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize to fixed size
        resized = cv2.resize(gray, self.resize_shape)
        
        # Extract HOG features
        features = hog(
            resized,
            orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            block_norm='L2-Hys',
            visualize=False,
            feature_vector=True
        )
        
        # Normalize the feature vector (L2 normalization)
        features = self._normalize(features)
        
        return features
    
    def extract_batch(self, image_paths: list) -> np.ndarray:
        """
        Extract features from multiple images.
        
        Args:
            image_paths: List of paths to image files
            
        Returns:
            Feature matrix of shape (n_images, feature_dim)
        """
        features_list = []
        for img_path in image_paths:
            try:
                features = self.extract(img_path)
                features_list.append(features)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                # Add zero vector for failed images
                features_list.append(np.zeros(self.feature_dim))
        
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
        return "hog_features"
    
    def get_feature_dim(self) -> int:
        """Return the dimensionality of the feature vector."""
        return self.feature_dim
