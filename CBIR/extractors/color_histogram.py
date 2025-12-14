"""
Color Histogram Feature Extractor

This extractor computes color histograms in HSV color space.
HSV is more robust to lighting changes than RGB.
"""

import cv2
import numpy as np
from typing import Union
from pathlib import Path


class ColorHistogramExtractor:
    """
    Extracts color histogram features from images using HSV color space.
    
    The histogram is computed for each channel (H, S, V) and concatenated
    into a single feature vector.
    """
    
    def __init__(self, h_bins: int = 50, s_bins: int = 60, v_bins: int = 60):
        """
        Initialize the Color Histogram extractor.
        
        Args:
            h_bins: Number of bins for Hue channel (0-180 in OpenCV)
            s_bins: Number of bins for Saturation channel (0-255)
            v_bins: Number of bins for Value channel (0-255)
        """
        self.h_bins = h_bins
        self.s_bins = s_bins
        self.v_bins = v_bins
        self.feature_dim = h_bins + s_bins + v_bins
        
    def extract(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Extract color histogram features from an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Normalized feature vector of shape (feature_dim,)
        """
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Compute histogram for each channel
        h_hist = cv2.calcHist([hsv], [0], None, [self.h_bins], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [self.s_bins], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [self.v_bins], [0, 256])
        
        # Concatenate histograms
        features = np.concatenate([
            h_hist.flatten(),
            s_hist.flatten(),
            v_hist.flatten()
        ])
        
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
        return "color_histogram"
    
    def get_feature_dim(self) -> int:
        """Return the dimensionality of the feature vector."""
        return self.feature_dim
