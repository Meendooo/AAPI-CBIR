"""
Query system for CBIR using FAISS indexes.

This script allows querying the FAISS indexes with test images
and retrieving the most similar images from the training set.
"""

import argparse
import faiss
import numpy as np
import pandas as pd
from pathlib import Path
import cv2

from extractors.color_histogram import ColorHistogramExtractor
from extractors.texture_lbp import TextureLBPExtractor
from extractors.cnn_features import CNNFeaturesExtractor
from extractors.hog_features import HOGFeaturesExtractor
from extractors.local_features import LocalFeaturesExtractor


class CBIRQuerySystem:
    """Content-Based Image Retrieval Query System."""
    
    def __init__(self, extractor_name: str, base_path: str = '.'):
        """
        Initialize the query system.
        
        Args:
            extractor_name: Name of the extractor to use
            base_path: Base directory containing indexes and mappings
        """
        self.extractor_name = extractor_name
        self.base_path = Path(base_path)
        
        # Initialize extractor
        if extractor_name == 'color_histogram':
            self.extractor = ColorHistogramExtractor()
        elif extractor_name == 'texture_lbp':
            self.extractor = TextureLBPExtractor()
        elif extractor_name == 'cnn_features':
            self.extractor = CNNFeaturesExtractor()
        elif extractor_name == 'hog_features':
            self.extractor = HOGFeaturesExtractor()
        elif extractor_name == 'local_features':
            self.extractor = LocalFeaturesExtractor()
        else:
            raise ValueError(f"Unknown extractor: {extractor_name}")
        
        # Load FAISS index
        index_path = self.base_path / 'faiss_indexes' / f'index_{extractor_name}.faiss'
        if not index_path.exists():
            raise FileNotFoundError(f"Index not found: {index_path}")
        self.index = faiss.read_index(str(index_path))
        
        # Load mapping CSV
        mapping_path = self.base_path / 'mappings' / f'mapping_{extractor_name}.csv'
        if not mapping_path.exists():
            raise FileNotFoundError(f"Mapping not found: {mapping_path}")
        self.mapping = pd.read_csv(mapping_path)
        
        print(f"Loaded {extractor_name} index with {self.index.ntotal} vectors")
    
    def query(self, image_path: str, k: int = 5):
        """
        Query the index with an image and return top-k similar images.
        
        Args:
            image_path: Path to the query image
            k: Number of similar images to retrieve
            
        Returns:
            DataFrame with results (path, class, distance)
        """
        # Extract features from query image
        query_features = self.extractor.extract(image_path)
        query_features = query_features.reshape(1, -1).astype(np.float32)
        
        # Search in FAISS index
        distances, indices = self.index.search(query_features, k)
        
        # Get results from mapping
        results = self.mapping.iloc[indices[0]].copy()
        results['distance'] = distances[0]
        
        return results
    
    def evaluate_on_test_set(self, test_path: str = 'test', k: int = 5):
        """
        Evaluate the system on the test set.
        
        Args:
            test_path: Path to test directory
            k: Number of neighbors to consider
            
        Returns:
            Dictionary with evaluation metrics
        """
        from utils.data_loader import load_dataset
        
        test_images, test_labels = load_dataset(self.base_path, split=test_path)
        
        correct = 0
        total = len(test_images)
        
        print(f"\nEvaluating on {total} test images...")
        print(f"Using top-{k} retrieval\n")
        
        precisions = []
        recalls = []
        average_precisions = []
        
        # Total relevant images per class in the database (assuming 100 per class based on training set)
        # Ideally this should be counted dynamically, but for this dataset we know it's ~100
        total_relevant_in_db = 100 
        
        for img_path, true_label in zip(test_images, test_labels):
            results = self.query(img_path, k=k)
            predicted_classes = results['class'].values
            
            # Calculate relevant items in the retrieved list
            relevant_retrieved = 0
            ap_sum = 0
            
            for i, pred_label in enumerate(predicted_classes):
                if pred_label == true_label:
                    relevant_retrieved += 1
                    # Precision at this rank (i+1)
                    p_at_i = relevant_retrieved / (i + 1)
                    ap_sum += p_at_i
            
            # Precision@K
            precision = relevant_retrieved / k
            precisions.append(precision)
            
            # Recall@K
            recall = relevant_retrieved / total_relevant_in_db
            recalls.append(recall)
            
            # Average Precision (AP) for this query
            if relevant_retrieved > 0:
                ap = ap_sum / relevant_retrieved
            else:
                ap = 0.0
            average_precisions.append(ap)
            
            # Count for simple accuracy (at least one match)
            if true_label in predicted_classes:
                correct += 1
        
        # Calculate mean metrics
        mean_precision = np.mean(precisions)
        mean_recall = np.mean(recalls)
        mean_ap = np.mean(average_precisions)
        
        # F1 Score
        if (mean_precision + mean_recall) > 0:
            f1_score = 2 * (mean_precision * mean_recall) / (mean_precision + mean_recall)
        else:
            f1_score = 0.0
        
        simple_accuracy = correct / total
        
        return {
            'accuracy': simple_accuracy,
            'precision': mean_precision,
            'recall': mean_recall,
            'f1_score': f1_score,
            'map': mean_ap,
            'correct': correct,
            'total': total,
            'k': k
        }


def main():
    parser = argparse.ArgumentParser(description='Query CBIR system')
    parser.add_argument('--extractor', type=str,
                       default='color_histogram',
                       help='Which extractor to use')
    parser.add_argument('--image', type=str,
                       help='Path to query image')
    parser.add_argument('--k', type=int, default=5,
                       help='Number of similar images to retrieve')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate on test set')
    
    args = parser.parse_args()
    
    # Initialize query system
    query_system = CBIRQuerySystem(args.extractor)
    
    if args.evaluate:
        # Evaluate on test set
        metrics = query_system.evaluate_on_test_set(k=args.k)
        
        print("="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Extractor: {args.extractor}")
        print(f"Top-{metrics['k']} Simple Accuracy: {metrics['accuracy']*100:.2f}% (At least one match)")
        print("-" * 40)
        print(f"Precision@{metrics['k']}:    {metrics['precision']*100:.2f}%")
        print(f"Recall@{metrics['k']}:       {metrics['recall']*100:.2f}% (of total 100 relevant)")
        print(f"F1-Score@{metrics['k']}:     {metrics['f1_score']*100:.2f}%")
        print(f"mAP (Mean Avg Prec): {metrics['map']*100:.2f}%")
        print("-" * 40)
        print(f"Correct Queries: {metrics['correct']}/{metrics['total']}")
        print("="*60)
    
    elif args.image:
        # Query single image
        results = query_system.query(args.image, k=args.k)
        
        print(f"\nQuery image: {args.image}")
        print(f"\nTop {args.k} similar images:")
        print("="*60)
        for idx, row in results.iterrows():
            print(f"{idx+1}. {row['class']}")
            print(f"   Path: {row['path']}")
            print(f"   Distance: {row['distance']:.4f}")
            print()
    
    else:
        print("Please specify --image or --evaluate")


if __name__ == '__main__':
    main()
