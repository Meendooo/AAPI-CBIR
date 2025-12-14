"""
Test script for the Texture LBP extractor.
"""

from extractors.texture_lbp import TextureLBPExtractor
from utils.data_loader import load_dataset
import numpy as np


def test_lbp_extractor():
    """Test the Texture LBP extractor."""
    
    print("="*60)
    print("Testing Texture LBP Extractor")
    print("="*60)
    
    # Initialize extractor
    print("\n1. Initializing extractor...")
    extractor = TextureLBPExtractor()
    print(f"   Extractor: {extractor.get_name()}")
    print(f"   Feature dimension: {extractor.get_feature_dim()}")
    print(f"   Radius: {extractor.radius}")
    print(f"   Points: {extractor.n_points}")
    print(f"   Method: {extractor.method}")
    
    # Load a few sample images
    print("\n2. Loading sample images...")
    train_paths, train_labels = load_dataset('.', split='train')
    
    # Test on first 5 images
    print("\n3. Testing feature extraction...")
    for i in range(min(5, len(train_paths))):
        img_path = train_paths[i]
        label = train_labels[i]
        
        print(f"\n   Image {i+1}: {label}")
        print(f"   Path: {img_path}")
        
        try:
            features = extractor.extract(img_path)
            print(f"   [OK] Features extracted successfully")
            print(f"   Shape: {features.shape}")
            print(f"   L2 norm: {np.linalg.norm(features):.6f} (should be ~1.0)")
            print(f"   Min value: {features.min():.6f}")
            print(f"   Max value: {features.max():.6f}")
            print(f"   Mean value: {features.mean():.6f}")
        except Exception as e:
            print(f"   [ERROR] Error: {e}")
    
    # Test batch extraction
    print("\n4. Testing batch extraction...")
    sample_paths = train_paths[:10]
    features_batch = extractor.extract_batch(sample_paths)
    print(f"   Batch shape: {features_batch.shape}")
    print(f"   Expected: (10, {extractor.get_feature_dim()})")
    
    print("\n" + "="*60)
    print("Testing completed!")
    print("="*60)


if __name__ == '__main__':
    test_lbp_extractor()
