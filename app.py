from typing import List

import matplotlib.pyplot as plt
import nltk
import numpy as np
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

def visualize_chunking_process(text: str, chunker: StatisticalChunker):
    """Visualize the chunking process step by step."""
    # Step 1: Split into sentences
    sentences = split_into_sentences(text)
    
    # Create a figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Step 2: Encode sentences and calculate similarities
    encoded_sentences = chunker._encode_documents(sentences)
    similarities = chunker._calculate_similarity_scores(encoded_sentences)
    
    # Plot similarity scores
    ax1.plot(similarities, marker='o', linestyle='-', color='b')
    ax1.set_title('Similarity Scores Between Sentences')
    ax1.set_xlabel('Sentence Index')
    ax1.set_ylabel('Similarity Score')
    ax1.grid(True)
    
    # Step 3: Create chunks
    chunks = chunker._chunk(sentences)
    
    # Plot chunk sizes
    chunk_sizes = [len(chunk.splits) for chunk in chunks]
    ax2.bar(range(len(chunk_sizes)), chunk_sizes, color='lightblue')
    ax2.set_title('Number of Sentences in Each Chunk')
    ax2.set_xlabel('Chunk Index')
    ax2.set_ylabel('Number of Sentences')
    ax2.grid(True)
    
    # Add chunk size labels
    for i, size in enumerate(chunk_sizes):
        ax2.text(i, size, str(size), ha='center', va='bottom')
    
    plt.tight_layout()
    return fig, chunks

def main():
    st.title("StatisticalChunker Visualization")
    
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
        help="Enter the text you want to chunk. The text will be split into sentences and then chunked based on semantic similarity."
    )
    
    # Chunker parameters
    st.subheader("Chunker Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        window_size = st.slider(
            "Window Size",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of previous sentences to consider for similarity calculation"
        )
    
    with col2:
        min_tokens = st.slider(
            "Minimum Tokens per Chunk",
            min_value=50,
            max_value=200,
            value=100,
            help="Minimum number of tokens in each chunk"
        )
    
    # Update chunker parameters
    chunker.window_size = window_size
    chunker.min_split_tokens = min_tokens
    
    if st.button("Create Chunks"):
        if not text:
            st.warning("Please enter some text first!")
            return
        
        with st.spinner("Processing..."):
            # Visualize the chunking process
            fig, chunks = visualize_chunking_process(text, chunker)
            st.pyplot(fig)
            
            # Display chunks
            st.subheader("Generated Chunks")
            for i, chunk in enumerate(chunks):
                with st.expander(f"Chunk {i+1} (Score: {chunk.triggered_score if chunk.triggered_score else 'N/A'})"):
                    st.write(" ".join(chunk.splits))
            
            # Display statistics
            if chunker.enable_statistics:
                st.subheader("Chunking Statistics")
                st.text(str(chunker.statistics))

if __name__ == "__main__":
    main()