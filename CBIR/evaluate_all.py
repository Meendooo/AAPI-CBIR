import pandas as pd
import numpy as np
import json
from query_system import CBIRQuerySystem
import os

def evaluate_all():
    extractors = [
        'hog_features',
        'cnn_features',
        'color_histogram',
        'local_features',
        'texture_lbp'
    ]
    
    results = []
    
    print("Starting comprehensive evaluation of all 5 extractors...")
    print("="*80)
    
    for extractor_name in extractors:
        print(f"\nEvaluating {extractor_name}...")
        try:
            qs = CBIRQuerySystem(extractor_name)
            metrics = qs.evaluate_on_test_set(k=5)
            
            metrics['extractor'] = extractor_name
            results.append(metrics)
            
            print(f"  -> Accuracy: {metrics['accuracy']*100:.2f}%")
            print(f"  -> mAP:      {metrics['map']*100:.2f}%")
            
        except Exception as e:
            print(f"  -> Failed: {e}")
    
    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)
    
    # Create DataFrame for nice display
    df = pd.DataFrame(results)
    
    # Reorder columns
    cols = ['extractor', 'accuracy', 'precision', 'recall', 'f1_score', 'map']
    df = df[cols]
    
    # Rename for display
    df.columns = ['Extractor', 'Accuracy', 'Precision@5', 'Recall@5', 'F1-Score@5', 'mAP']
    
    # Sort by mAP (usually the best metric)
    df = df.sort_values('mAP', ascending=False)
    
    print(df.to_string(index=False, float_format=lambda x: "{:.2f}%".format(x*100)))
    
    # Save to JSON for the UI
    output_path = 'evaluation_results.json'
    df.to_json(output_path, orient='records', indent=4)
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    evaluate_all()
