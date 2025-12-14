"""
Streamlit User Interface for CBIR System.

This application provides a web-based interface to interact with the Content-Based
Image Retrieval system. It allows users to:
1. Select from 5 different feature extractors
2. Upload and crop query images
3. View retrieval results with similarity metrics
4. Compare performance across different methods
"""

import time
import faiss
import pathlib
from PIL import Image
import numpy as np
import pandas as pd
import os
import warnings

# Suppress warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
warnings.filterwarnings('ignore')

import streamlit as st
from streamlit_cropper import st_cropper

from extractors.color_histogram import ColorHistogramExtractor
from extractors.texture_lbp import TextureLBPExtractor
from extractors.cnn_features import CNNFeaturesExtractor
from extractors.hog_features import HOGFeaturesExtractor
from extractors.local_features import LocalFeaturesExtractor

st.set_page_config(layout="wide", page_title="CBIR Card Search")

# Get the directory where app.py is located
SCRIPT_DIR = pathlib.Path(__file__).parent.absolute()

# Path configurations - use absolute paths
FAISS_PATH = SCRIPT_DIR / 'faiss_indexes'
MAPPINGS_PATH = SCRIPT_DIR / 'mappings'


@st.cache_resource
def load_extractor(extractor_name):
    """Load and cache the feature extractor."""
    if extractor_name == 'Color Histogram':
        return ColorHistogramExtractor()
    elif extractor_name == 'Texture LBP':
        return TextureLBPExtractor()
    elif extractor_name == 'CNN Features (ResNet50)':
        return CNNFeaturesExtractor()
    elif extractor_name == 'HOG':
        return HOGFeaturesExtractor()
    elif extractor_name == 'Local Features (ORB)':
        return LocalFeaturesExtractor()


@st.cache_resource
def load_faiss_index(extractor_key):
    """Load and cache FAISS index."""
    index_path = FAISS_PATH / f'index_{extractor_key}.faiss'
    
    # Verify file exists
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found at: {index_path}")
    
    # Load into memory first to avoid Windows path encoding issues in FAISS C++
    # The path contains special characters (º) which FAISS C++ reader fails to handle
    with open(index_path, 'rb') as f:
        data = f.read()
    
    # Deserialize from bytes
    vector = np.frombuffer(data, dtype='uint8')
    return faiss.deserialize_index(vector)


@st.cache_data
def load_mapping(extractor_key):
    """Load and cache the mapping CSV."""
    mapping_path = MAPPINGS_PATH / f'mapping_{extractor_key}.csv'
    
    # Verify file exists
    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping CSV not found at: {mapping_path}")
    
    return pd.read_csv(str(mapping_path.absolute()))


def get_extractor_key(extractor_name):
    """Convert display name to file key."""
    mapping = {
        'Color Histogram': 'color_histogram',
        'Texture LBP': 'texture_lbp',
        'CNN Features (ResNet50)': 'cnn_features',
        'HOG': 'hog_features',
        'Local Features (ORB)': 'local_features'
    }
    return mapping[extractor_name]


def retrieve_image(img_query, extractor_name, n_imgs=11):
    """Retrieve similar images using the selected extractor."""
    # Get extractor key
    extractor_key = get_extractor_key(extractor_name)
    
    # Load extractor, index, and mapping
    extractor = load_extractor(extractor_name)
    indexer = load_faiss_index(extractor_key)
    mapping_df = load_mapping(extractor_key)
    
    # Save temporary image
    temp_path = 'temp_query.jpg'
    img_query.save(temp_path)
    
    # Extract features
    features = extractor.extract(temp_path)
    vector = features.reshape(1, -1).astype(np.float32)
    
    # Search in FAISS
    distances, indices = indexer.search(vector, k=n_imgs)
    
    # Clean up
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    # Get image paths and classes
    results = mapping_df.iloc[indices[0]].copy()
    results['distance'] = distances[0]
    
    return results


def main():
    st.title('🃏 CBIR CARD IMAGE SEARCH')
    st.markdown('**Content-Based Image Retrieval System for Playing Cards**')
    
    col1, col2 = st.columns(2)

    with col1:
        st.header('🔍 QUERY')

        st.subheader('Choose Feature Extractor')
        option = st.selectbox(
            'Select the feature extraction method:',
            (
                'Color Histogram',
                'Texture LBP',
                'CNN Features (ResNet50)',
                'HOG',
                'Local Features (ORB)'
            )
        )
        
        # Show extractor info
        extractor_info = {
            'Color Histogram': '📊 82% accuracy - Fast, color-based',
            'Texture LBP': '🔲 76% accuracy - Texture patterns',
            'CNN Features (ResNet50)': '🧠 90% accuracy - Deep learning',
            'HOG': '⭐ 92% accuracy - Shape & edges (BEST)',
            'Local Features (ORB)': '🎯 80% accuracy - Keypoints'
        }
        st.info(extractor_info[option])

        st.subheader('Upload Image')
        img_file = st.file_uploader('Upload a card image:', type=['png', 'jpg', 'jpeg'])

        if img_file:
            img = Image.open(img_file)
            # Get a cropped image from the frontend
            cropped_img = st_cropper(img, realtime_update=True, box_color='#FF0004', aspect_ratio=None)
            
            # Manipulate cropped image at will
            st.write("**Preview:**")
            preview_img = cropped_img.copy()
            preview_img.thumbnail((150, 150))
            st.image(preview_img)

    with col2:
        st.header('📋 RESULTS')
        if img_file:
            st.markdown('**🔄 Retrieving similar images...**')
            start = time.time()

            results = retrieve_image(cropped_img, option, n_imgs=11)

            end = time.time()
            st.success(f'✅ **Completed in {end - start:.2f} seconds**')
            
            # Show top result info
            st.markdown(f"**Top Match:** {results.iloc[0]['class']}")
            st.markdown(f"**Distance:** {results.iloc[0].get('distance', 'N/A')}")

            # Display top 2 results in larger size
            col3, col4 = st.columns(2)

            with col3:
                st.markdown("**1st Match**")
                image_path = results.iloc[0]['path']
                image = Image.open(image_path)
                st.image(image, use_container_width=True)
                st.caption(f"Class: {results.iloc[0]['class']}")

            with col4:
                st.markdown("**2nd Match**")
                image_path = results.iloc[1]['path']
                image = Image.open(image_path)
                st.image(image, use_container_width=True)
                st.caption(f"Class: {results.iloc[1]['class']}")

            # Display remaining results in 3 columns
            st.markdown("**Other Similar Images:**")
            col5, col6, col7 = st.columns(3)

            with col5:
                for u in range(2, 11, 3):
                    if u < len(results):
                        image_path = results.iloc[u]['path']
                        image = Image.open(image_path)
                        st.image(image, use_container_width=True)
                        st.caption(f"{u+1}. {results.iloc[u]['class']}")

            with col6:
                for u in range(3, 11, 3):
                    if u < len(results):
                        image_path = results.iloc[u]['path']
                        image = Image.open(image_path)
                        st.image(image, use_container_width=True)
                        st.caption(f"{u+1}. {results.iloc[u]['class']}")

            with col7:
                for u in range(4, 11, 3):
                    if u < len(results):
                        image_path = results.iloc[u]['path']
                        image = Image.open(image_path)
                        st.image(image, use_container_width=True)
                        st.caption(f"{u+1}. {results.iloc[u]['class']}")
    
    # Sidebar with statistics
    with st.sidebar:
        st.header("📊 System Statistics")
        
        # Load evaluation results if available
        eval_path = SCRIPT_DIR / 'evaluation_results.json'
        if eval_path.exists():
            try:
                import json
                with open(eval_path, 'r') as f:
                    metrics_data = json.load(f)
                
                st.subheader("🏆 Performance Leaderboard")
                
                # Create a clean table for the sidebar
                df_metrics = pd.DataFrame(metrics_data)
                df_metrics = df_metrics[['Extractor', 'Accuracy', 'Precision@5', 'Recall@5', 'mAP']]
                
                # Format percentages
                df_metrics['Accuracy'] = df_metrics['Accuracy'].apply(lambda x: f"{x*100:.1f}%")
                df_metrics['Precision@5'] = df_metrics['Precision@5'].apply(lambda x: f"{x*100:.1f}%")
                df_metrics['Recall@5'] = df_metrics['Recall@5'].apply(lambda x: f"{x*100:.1f}%")
                df_metrics['mAP'] = df_metrics['mAP'].apply(lambda x: f"{x*100:.1f}%")
                
                st.table(df_metrics)
                
                st.markdown("---")
                st.markdown("**Metric Definitions:**")
                st.caption("**Accuracy**: % queries with ≥1 match.")
                st.caption("**Precision**: % correct images in top-5.")
                st.caption("**Recall**: % total relevant images found.")
                st.caption("**mAP**: Rank quality (higher is better).")
                
            except Exception as e:
                st.error(f"Error loading metrics: {e}")
        else:
            st.markdown("""
            **Extractors Performance:**
            - 🥇 HOG: 92%
            - 🥈 CNN: 90%
            - 🥉 Color: 82%
            - Local: 80%
            - Texture: 76%
            """)
            
        st.markdown("---")
        st.markdown("""
        **Dataset:**
        - Training: 500 images
        - Test: 50 images
        - Classes: 5
        """)


if __name__ == '__main__':
    main()
