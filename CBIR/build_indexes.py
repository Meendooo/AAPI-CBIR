"""
Build FAISS indexes for all feature extractors.

This script extracts features from the training set and creates FAISS indexes
along with CSV mappings for image retrieval.
"""

import argparse
import faiss
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import os

from extractors.color_histogram import ColorHistogramExtractor
from extractors.texture_lbp import TextureLBPExtractor
from extractors.cnn_features import CNNFeaturesExtractor
from extractors.hog_features import HOGFeaturesExtractor
from extractors.local_features import LocalFeaturesExtractor
from utils.data_loader import load_dataset


def build_index_for_extractor(extractor, extractor_name: str, 
                              base_path: str, output_dir: str):
    """
    Build FAISS index for a specific extractor.
    
    Args:
        extractor: Feature extractor instance
        extractor_name: Name of the extractor (for file naming)
        base_path: Base directory containing train/test folders
        output_dir: Directory to save indexes and mappings
    """
    print(f"\n{'='*60}")
    print(f"Building index for: {extractor_name}")
    print(f"{'='*60}")
    
    # Load training data
    print("Loading training dataset...")
    train_paths, train_labels = load_dataset(base_path, split='train')
    print(f"Found {len(train_paths)} training images")
    
    # Train vocabulary if using local features
    if extractor_name == 'local_features':
        print("\nTraining visual vocabulary for local features...")
        extractor.train_vocabulary(train_paths)
        print("Vocabulary training completed.\n")
    
    # Extract features
    print("Extracting features...")
    features_list = []
    valid_paths = []
    valid_labels = []
    
    for img_path, label in tqdm(zip(train_paths, train_labels), 
                                total=len(train_paths)):
        try:
            features = extractor.extract(img_path)
            features_list.append(features)
            valid_paths.append(img_path)
            valid_labels.append(label)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Convert to numpy array
    features_matrix = np.array(features_list, dtype=np.float32)
    print(f"Feature matrix shape: {features_matrix.shape}")
    
    # Create FAISS index
    print("Creating FAISS index...")
    dimension = features_matrix.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(features_matrix)
    print(f"Index created with {index.ntotal} vectors")
    
    # Create output directories
    os.makedirs(f"{output_dir}/faiss_indexes", exist_ok=True)
    os.makedirs(f"{output_dir}/mappings", exist_ok=True)
    
    # Save FAISS index
    index_path = f"{output_dir}/faiss_indexes/index_{extractor_name}.faiss"
    faiss.write_index(index, index_path)
    print(f"FAISS index saved to: {index_path}")
    
    # Create and save CSV mapping
    mapping_df = pd.DataFrame({
        'idx': range(len(valid_paths)),
        'path': valid_paths,
        'class': valid_labels
    })
    
    mapping_path = f"{output_dir}/mappings/mapping_{extractor_name}.csv"
    mapping_df.to_csv(mapping_path, index=False)
    print(f"Mapping CSV saved to: {mapping_path}")
    
    # Print statistics
    print(f"\nStatistics:")
    print(f"  Total vectors: {len(features_list)}")
    print(f"  Feature dimension: {dimension}")
    print(f"  Classes: {mapping_df['class'].nunique()}")
    print(f"  Class distribution:")
    for class_name, count in mapping_df['class'].value_counts().items():
        print(f"    {class_name}: {count}")


def main():
    parser = argparse.ArgumentParser(description='Build FAISS indexes')
    parser.add_argument('--base-path', type=str, 
                       default='.',
                       help='Base directory containing train/test folders')
    parser.add_argument('--output-dir', type=str,
                       default='.',
                       help='Output directory for indexes and mappings')
    parser.add_argument('--extractor', type=str,
                       default='color_histogram',
                       choices=['color_histogram', 'texture_lbp', 'cnn_features', 'hog_features', 'local_features'],
                       help='Which extractor to use')
    
    args = parser.parse_args()
    
    # Initialize extractor
    if args.extractor == 'color_histogram':
        extractor = ColorHistogramExtractor()
    elif args.extractor == 'texture_lbp':
        extractor = TextureLBPExtractor()
    elif args.extractor == 'cnn_features':
        extractor = CNNFeaturesExtractor()
    elif args.extractor == 'hog_features':
        extractor = HOGFeaturesExtractor()
    elif args.extractor == 'local_features':
        extractor = LocalFeaturesExtractor()
    else:
        raise ValueError(f"Unknown extractor: {args.extractor}")
    
    # Build index
    build_index_for_extractor(
        extractor=extractor,
        extractor_name=args.extractor,
        base_path=args.base_path,
        output_dir=args.output_dir
    )
    
    print(f"\n{'='*60}")
    print("Index building completed successfully!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
