"""
Data loading utilities for CBIR system.
"""

from pathlib import Path
from typing import Tuple, List
import os


def load_dataset(base_path: str, split: str = 'train') -> Tuple[List[str], List[str]]:
    """
    Load image paths and labels from the dataset.
    
    Args:
        base_path: Base directory containing train/test folders
        split: Either 'train' or 'test'
        
    Returns:
        Tuple of (image_paths, labels)
    """
    base_path = Path(base_path)
    split_path = base_path / split
    
    if not split_path.exists():
        raise ValueError(f"Split path does not exist: {split_path}")
    
    image_paths = []
    labels = []
    
    # Get all class directories
    class_dirs = sorted([d for d in split_path.iterdir() if d.is_dir()])
    
    for class_dir in class_dirs:
        class_name = class_dir.name
        
        # Get all images in this class
        image_files = sorted(list(class_dir.glob('*.jpg')) + 
                           list(class_dir.glob('*.png')) + 
                           list(class_dir.glob('*.jpeg')))
        
        for img_path in image_files:
            image_paths.append(str(img_path))
            labels.append(class_name)
    
    return image_paths, labels


def verify_dataset(base_path: str) -> dict:
    """
    Verify the dataset structure and print statistics.
    
    Args:
        base_path: Base directory containing train/test folders
        
    Returns:
        Dictionary with dataset statistics
    """
    base_path = Path(base_path)
    stats = {
        'train': {},
        'test': {}
    }
    
    for split in ['train', 'test']:
        split_path = base_path / split
        
        if not split_path.exists():
            print(f"WARNING: {split} directory not found at {split_path}")
            continue
        
        class_dirs = sorted([d for d in split_path.iterdir() if d.is_dir()])
        
        print(f"\n{split.upper()} SET:")
        print("-" * 50)
        
        total_images = 0
        for class_dir in class_dirs:
            class_name = class_dir.name
            image_files = list(class_dir.glob('*.jpg')) + \
                         list(class_dir.glob('*.png')) + \
                         list(class_dir.glob('*.jpeg'))
            
            num_images = len(image_files)
            total_images += num_images
            stats[split][class_name] = num_images
            
            print(f"  {class_name}: {num_images} images")
        
        print(f"\n  Total: {total_images} images")
        print(f"  Classes: {len(class_dirs)}")
    
    return stats


def get_class_mapping(base_path: str) -> dict:
    """
    Get a mapping from class names to integer indices.
    
    Args:
        base_path: Base directory containing train/test folders
        
    Returns:
        Dictionary mapping class names to indices
    """
    base_path = Path(base_path)
    train_path = base_path / 'train'
    
    class_dirs = sorted([d.name for d in train_path.iterdir() if d.is_dir()])
    
    return {class_name: idx for idx, class_name in enumerate(class_dirs)}
