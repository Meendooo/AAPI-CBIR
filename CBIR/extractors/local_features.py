"""
Local Features Extractor using ORB (Oriented FAST and Rotated BRIEF)

ORB detects keypoints and computes local descriptors. Since this generates
multiple vectors per image, we use Bag of Visual Words (BoVW) to aggregate
them into a single fixed-length feature vector.
"""

import cv2
import numpy as np
from typing import Union, List
from pathlib import Path
import pickle
from sklearn.cluster import MiniBatchKMeans


class LocalFeaturesExtractor:
    """
    Extracts local features using ORB and aggregates them using Bag of Visual Words.
    
    ORB is rotation invariant and scale invariant, making it robust for
    various transformations.
    """
    
    def __init__(self, 
                 n_features: int = 500,
                 vocabulary_size: int = 200,
                 vocabulary_path: str = None):
        """
        Initialize the Local Features extractor.
        
        Args:
            n_features: Maximum number of keypoints to detect (default: 500)
            vocabulary_size: Size of visual vocabulary (default: 200)
            vocabulary_path: Path to save/load vocabulary (default: None)
        """
        self.n_features = n_features
        self.vocabulary_size = vocabulary_size
        self.vocabulary_path = vocabulary_path or './vocabulary_orb.pkl'
        self.feature_dim = vocabulary_size
        
        # Initialize ORB detector
        self.orb = cv2.ORB_create(nfeatures=n_features)
        
        # Visual vocabulary (will be trained)
        self.vocabulary = None
        self.kmeans = None
        
    def train_vocabulary(self, image_paths: List[str], sample_size: int = None):
        """
        Train the visual vocabulary using k-means clustering on descriptors.
        
        Args:
            image_paths: List of training image paths
            sample_size: Number of images to sample (None = use all)
        """
        print(f"Training visual vocabulary with {self.vocabulary_size} words...")
        
        # Sample images if needed
        if sample_size and sample_size < len(image_paths):
            import random
            image_paths = random.sample(image_paths, sample_size)
        
        # Extract all descriptors
        all_descriptors = []
        for img_path in image_paths:
            try:
                image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue
                
                keypoints, descriptors = self.orb.detectAndCompute(image, None)
                
                if descriptors is not None and len(descriptors) > 0:
                    all_descriptors.append(descriptors)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        # Concatenate all descriptors
        all_descriptors = np.vstack(all_descriptors)
        print(f"Collected {len(all_descriptors)} descriptors from {len(image_paths)} images")
        
        # Train k-means
        print("Training k-means clustering...")
        self.kmeans = MiniBatchKMeans(
            n_clusters=self.vocabulary_size,
            random_state=42,
            batch_size=1000,
            max_iter=100
        )
        self.kmeans.fit(all_descriptors.astype(np.float32))
        self.vocabulary = self.kmeans.cluster_centers_
        
        # Save vocabulary
        self._save_vocabulary()
        print(f"Vocabulary trained and saved to {self.vocabulary_path}")
    
    def _save_vocabulary(self):
        """Save the trained vocabulary to disk."""
        with open(self.vocabulary_path, 'wb') as f:
            pickle.dump({
                'vocabulary': self.vocabulary,
                'kmeans': self.kmeans,
                'vocabulary_size': self.vocabulary_size
            }, f)
    
    def _load_vocabulary(self):
        """Load the trained vocabulary from disk."""
        try:
            with open(self.vocabulary_path, 'rb') as f:
                data = pickle.load(f)
                self.vocabulary = data['vocabulary']
                self.kmeans = data['kmeans']
                self.vocabulary_size = data['vocabulary_size']
                self.feature_dim = self.vocabulary_size
            print(f"Vocabulary loaded from {self.vocabulary_path}")
            return True
        except FileNotFoundError:
            return False
    
    def extract(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Extract BoVW features from an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Normalized feature vector (histogram of visual words)
        """
        # Load vocabulary if not loaded
        if self.vocabulary is None:
            if not self._load_vocabulary():
                raise ValueError("Vocabulary not trained. Call train_vocabulary() first.")
        
        # Read image
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Detect keypoints and compute descriptors
        keypoints, descriptors = self.orb.detectAndCompute(image, None)
        
        # Create histogram of visual words
        histogram = np.zeros(self.vocabulary_size, dtype=np.float32)
        
        if descriptors is not None and len(descriptors) > 0:
            # Assign each descriptor to nearest visual word
            labels = self.kmeans.predict(descriptors.astype(np.float32))
            
            # Build histogram
            for label in labels:
                histogram[label] += 1
        
        # Normalize the histogram (L2 normalization)
        histogram = self._normalize(histogram)
        
        return histogram
    
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
        return "local_features"
    
    def get_feature_dim(self) -> int:
        """Return the dimensionality of the feature vector."""
        return self.feature_dim
