from typing import List, Tuple

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import streamlit as st
from nltk.tokenize import sent_tokenize

# from semantic_router.encoders import OpenAIEncoder
from giga_encoder import GigaChatEncoder
from semantic_chunkers.chunkers.statistical import StatisticalChunker

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using NLTK."""
    return sent_tokenize(text)

def visualize_step(
    text: str,
    chunker: StatisticalChunker,
    step: int,
    batch_size: int = 64
) -> Tuple[plt.Figure, List[str], List[float], List[int]]:
    """Visualize a specific step of the chunking process."""
    # Step 1: Initial split using chunker's splitter
    if step == 1:
        splits = chunker._split(text)
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.axis('off')
        ax.text(0.1, 0.5, f"Initial splits ({len(splits)}):\n\n" + "\n".join(splits), 
                wrap=True, fontsize=10)
        return fig, splits, [], []
    
    # Step 2: Encode documents
    elif step == 2:
        splits = chunker._split(text)
        encoded_splits = chunker._encode_documents(splits)
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.axis('off')
        ax.text(0.1, 0.5, f"Encoded {len(encoded_splits)} splits into embeddings", 
                wrap=True, fontsize=10)
        return fig, splits, [], []
    
    # Step 3: Calculate similarity scores
    elif step == 3:
        splits = chunker._split(text)
        encoded_splits = chunker._encode_documents(splits)
        similarities = chunker._calculate_similarity_scores(encoded_splits)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(similarities, marker='o', linestyle='-', color='b')
        ax.set_title('Similarity Scores Between Splits')
        ax.set_xlabel('Split Index')
        ax.set_ylabel('Similarity Score')
        ax.grid(True)
        
        # Add similarity scores as text
        for i, score in enumerate(similarities):
            ax.text(i, score, f'{score:.2f}', ha='center', va='bottom')
        
        return fig, splits, similarities, []
    
    # Step 4: Find optimal threshold
    elif step == 4:
        splits = chunker._split(text)
        encoded_splits = chunker._encode_documents(splits)
        similarities = chunker._calculate_similarity_scores(encoded_splits)
        
        if chunker.dynamic_threshold:
            threshold = chunker._find_optimal_threshold(splits, similarities)
        else:
            threshold = chunker.encoder.score_threshold or chunker.DEFAULT_THRESHOLD
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(similarities, marker='o', linestyle='-', color='b')
        ax.axhline(y=threshold, color='r', linestyle='--', 
                  label=f'Threshold: {threshold:.2f}')
        ax.set_title('Similarity Scores with Threshold')
        ax.set_xlabel('Split Index')
        ax.set_ylabel('Similarity Score')
        ax.grid(True)
        ax.legend()
        
        return fig, splits, similarities, []
    
    # Step 5: Find split indices
    elif step == 5:
        splits = chunker._split(text)
        encoded_splits = chunker._encode_documents(splits)
        similarities = chunker._calculate_similarity_scores(encoded_splits)
        
        if chunker.dynamic_threshold:
            threshold = chunker._find_optimal_threshold(splits, similarities)
        else:
            threshold = chunker.encoder.score_threshold or chunker.DEFAULT_THRESHOLD
            
        split_indices = chunker._find_split_indices(similarities, threshold)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(similarities, marker='o', linestyle='-', color='b')
        ax.axhline(y=threshold, color='r', linestyle='--', 
                  label=f'Threshold: {threshold:.2f}')
        
        # Mark split points
        for idx in split_indices:
            ax.axvline(x=idx-1, color='g', linestyle=':', alpha=0.5)
            ax.text(idx-1, threshold, f'Split {idx}', 
                   rotation=90, va='bottom', ha='center')
        
        ax.set_title('Split Points Based on Similarity')
        ax.set_xlabel('Split Index')
        ax.set_ylabel('Similarity Score')
        ax.grid(True)
        ax.legend()
        
        return fig, splits, similarities, split_indices
    
    # Step 6: Final chunks
    else:
        splits = chunker._split(text)
        chunks = chunker._chunk(splits, batch_size=batch_size)
        
        # Create visualization of chunks
        fig, ax = plt.subplots(figsize=(12, 6))
        chunk_sizes = [len(chunk.splits) for chunk in chunks]
        ax.bar(range(len(chunk_sizes)), chunk_sizes, color='lightblue')
        ax.set_title('Final Chunks')
        ax.set_xlabel('Chunk Index')
        ax.set_ylabel('Number of Splits')
        ax.grid(True)
        
        # Add chunk sizes as text
        for i, size in enumerate(chunk_sizes):
            ax.text(i, size, str(size), ha='center', va='bottom')
        
        return fig, splits, [], []

def main():
    st.title("StatisticalChunker Step-by-Step Visualization")
    
    # Initialize the encoder and chunker
    with st.spinner("Initializing encoder..."):
        encoder = GigaChatEncoder(name="EmbeddingsGigaR")
        chunker = StatisticalChunker(
            encoder=encoder,
            window_size=5,
            min_split_tokens=100,
            max_split_tokens=300,
            plot_chunks=True,
            enable_statistics=True
        )
    
    # Text input
    text = st.text_area(
        "Enter your text here:",
        height=200,
        help="Enter the text you want to chunk."
    )
    
    # Chunker parameters
    st.subheader("Chunker Parameters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        window_size = st.slider(
            "Window Size",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of previous splits to consider for similarity"
        )
    
    with col2:
        min_tokens = st.slider(
            "Minimum Tokens",
            min_value=50,
            max_value=200,
            value=100,
            help="Minimum tokens per chunk"
        )
    
    with col3:
        dynamic_threshold = st.checkbox(
            "Dynamic Threshold",
            value=True,
            help="Automatically find optimal threshold"
        )
    
    # Update chunker parameters
    chunker.window_size = window_size
    chunker.min_split_tokens = min_tokens
    chunker.dynamic_threshold = dynamic_threshold
    
    if not text:
        st.warning("Please enter some text first!")
        return
    
    # Step navigation
    st.subheader("Chunking Process")
    current_step = st.session_state.get('current_step', 1)
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("Previous Step") and current_step > 1:
            current_step -= 1
            st.session_state.current_step = current_step
        if st.button("Next Step") and current_step < 6:
            current_step += 1
            st.session_state.current_step = current_step
    
    with col2:
        st.progress(current_step / 6)
        st.text(f"Step {current_step}/6")
    
    # Show current step visualization
    with st.spinner("Processing..."):
        fig, splits, similarities, split_indices = visualize_step(
            text, chunker, current_step
        )
        st.pyplot(fig)
        
        # Show additional information based on current step
        if current_step == 1:
            st.write("Initial splits of the text:")
            for i, split in enumerate(splits):
                st.text(f"Split {i+1}: {split}")
        
        elif current_step == 3:
            st.write("Similarity scores between consecutive splits:")
            df = pd.DataFrame({
                'Split Index': range(1, len(similarities) + 1),
                'Similarity Score': similarities
            })
            st.dataframe(df)
        
        elif current_step == 5:
            st.write("Split points based on similarity threshold:")
            for idx in split_indices:
                st.text(f"Split point {idx}: {splits[idx-1]}")
        
        elif current_step == 6:
            st.write("Final chunks:")
            chunks = chunker._chunk(splits)
            for i, chunk in enumerate(chunks):
                with st.expander(f"Chunk {i+1} (Score: {chunk.triggered_score if chunk.triggered_score else 'N/A'})"):
                    st.write(" ".join(chunk.splits))
            
            if chunker.enable_statistics:
                st.subheader("Chunking Statistics")
                st.text(str(chunker.statistics))

if __name__ == "__main__":
    main()