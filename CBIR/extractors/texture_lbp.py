"""
Texture Feature Extractor using Local Binary Patterns (LBP)

LBP is a texture descriptor that is robust to illumination changes
and computationally efficient.
"""

import cv2
import numpy as np
from typing import Union
from pathlib import Path
from skimage.feature import local_binary_pattern


class TextureLBPExtractor:
    """
    Extracts texture features using Local Binary Patterns (LBP).
    
    LBP captures local texture information by comparing each pixel
    with its neighbors and creating a binary pattern.
    """
    
    def __init__(self, radius: int = 3, n_points: int = 24, method: str = 'uniform'):
        """
        Initialize the LBP extractor.
        
        Args:
            radius: Radius of circle for LBP (default: 3)
            n_points: Number of circularly symmetric neighbor points (default: 24)
            method: LBP method ('uniform', 'default', 'ror', 'var')
                   'uniform' is recommended for most cases
        """
        self.radius = radius
        self.n_points = n_points
        self.method = method
        
        # Calculate feature dimension based on method
        if method == 'uniform':
            self.feature_dim = n_points + 2  # uniform patterns + 2 bins
        else:
            self.feature_dim = 2 ** n_points
        
        # We'll use 3 channels (RGB) and concatenate histograms
        self.feature_dim = self.feature_dim * 3
        
    def extract(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Extract LBP texture features from an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Normalized feature vector
        """
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Convert to RGB (OpenCV loads as BGR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Extract LBP features for each channel
        features_list = []
        
        for channel in range(3):
            channel_img = image_rgb[:, :, channel]
            
            # Compute LBP
            lbp = local_binary_pattern(
                channel_img, 
                self.n_points, 
                self.radius, 
                method=self.method
            )
            
            # Compute histogram
            if self.method == 'uniform':
                n_bins = self.n_points + 2
            else:
                n_bins = 2 ** self.n_points
            
            hist, _ = np.histogram(
                lbp.ravel(),
                bins=n_bins,
                range=(0, n_bins),
                density=True
            )
            
            features_list.append(hist)
        
        # Concatenate histograms from all channels
        features = np.concatenate(features_list)
        
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
        return "texture_lbp"
    
    def get_feature_dim(self) -> int:
        """Return the dimensionality of the feature vector."""
        return self.feature_dim
