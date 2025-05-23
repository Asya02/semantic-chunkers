from typing import List, Tuple

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import streamlit as st
from nltk.tokenize import sent_tokenize

# from semantic_router.encoders import OpenAIEncoder
from giga_encoder import GigaChatEncoder
from semantic_chunkers.chunkers.consecutive import ConsecutiveChunker
from semantic_chunkers.chunkers.cumulative import CumulativeChunker
from semantic_chunkers.chunkers.statistical import StatisticalChunker

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using NLTK."""
    return sent_tokenize(text)

def visualize_consecutive_step(
    text: str,
    chunker: ConsecutiveChunker,
    step: int,
    batch_size: int = 64
) -> Tuple[plt.Figure, List[str], List[float], List[int]]:
    """Visualize a specific step of the consecutive chunking process."""
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
        encoded_splits = chunker.encoder(splits)
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.axis('off')
        ax.text(0.1, 0.5, f"Encoded {len(encoded_splits)} splits into embeddings", 
                wrap=True, fontsize=10)
        return fig, splits, [], []
    
    # Step 3: Calculate similarity matrix
    elif step == 3:
        splits = chunker._split(text)
        encoded_splits = chunker.encoder(splits)
        norm_embeds = encoded_splits / np.linalg.norm(encoded_splits, axis=1, keepdims=True)
        sim_matrix = np.matmul(norm_embeds, norm_embeds.T)
        
        # Get consecutive similarities
        consecutive_sims = [sim_matrix[i][i+1] for i in range(len(sim_matrix)-1)]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(consecutive_sims, marker='o', linestyle='-', color='b')
        ax.axhline(y=chunker.score_threshold, color='r', linestyle='--', 
                  label=f'Threshold: {chunker.score_threshold:.2f}')
        ax.set_title('Similarity Scores Between Consecutive Splits')
        ax.set_xlabel('Split Index')
        ax.set_ylabel('Similarity Score')
        ax.grid(True)
        ax.legend()
        
        # Add similarity scores as text
        for i, score in enumerate(consecutive_sims):
            ax.text(i, score, f'{score:.2f}', ha='center', va='bottom')
        
        return fig, splits, consecutive_sims, []
    
    # Step 4: Final chunks
    else:
        splits = chunker._split(text)
        chunks = chunker._chunk(splits)
        
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

def visualize_statistical_step(
    text: str,
    chunker: StatisticalChunker,
    step: int,
    batch_size: int = 64
) -> Tuple[plt.Figure, List[str], List[float], List[int]]:
    """Visualize a specific step of the statistical chunking process."""
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

def visualize_cumulative_step(
    text: str,
    chunker: CumulativeChunker,
    step: int,
    batch_size: int = 64
) -> Tuple[plt.Figure, List[str], List[float], List[int]]:
    """Visualize a specific step of the cumulative chunking process."""
    # Step 1: Initial split using chunker's splitter
    if step == 1:
        splits = chunker._split(text)
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.axis('off')
        ax.text(0.1, 0.5, f"Initial splits ({len(splits)}):\n\n" + "\n".join(splits), 
                wrap=True, fontsize=10)
        return fig, splits, [], []
    
    # Step 2: Show cumulative text building
    elif step == 2:
        splits = chunker._split(text)
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')
        
        # Create a text display showing accumulation
        y_pos = 0.95
        for i in range(len(splits)):
            # Show current step number
            ax.text(0.1, y_pos, f"Step {i+1}:", fontsize=12, weight='bold')
            y_pos -= 0.05
            
            # Show cumulative text with color highlighting
            if i == 0:
                # First split is shown in blue
                ax.text(0.15, y_pos, splits[0], 
                       color='blue', wrap=True, fontsize=10)
            else:
                # Show previous splits in gray
                prev_text = "\n".join(splits[:i])
                ax.text(0.15, y_pos, prev_text, 
                       color='gray', wrap=True, fontsize=10)
                y_pos -= 0.02
                # Show current split in blue
                ax.text(0.15, y_pos, splits[i], 
                       color='blue', wrap=True, fontsize=10)
            
            y_pos -= 0.05
            
            # Show next split if exists
            if i < len(splits) - 1:
                ax.text(0.15, y_pos, "Comparing with next split:", 
                       color='red', wrap=True, fontsize=10)
                y_pos -= 0.02
                ax.text(0.15, y_pos, splits[i+1], 
                       color='red', wrap=True, fontsize=10)
            
            y_pos -= 0.1  # Add space between steps
        
        # Add legend
        ax.text(0.1, 0.02, 
                "Legend:\nBlue = Current cumulative text\nGray = Previous text\nRed = Next split to compare with",
                fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
        
        return fig, splits, [], []
    
    # Step 3: Calculate and show similarities
    elif step == 3:
        splits = chunker._split(text)
        similarities = []
        
        for idx in range(len(splits) - 1):
            if idx == 0:
                curr_chunk_docs = splits[idx]
            else:
                curr_chunk_docs = "\n".join(splits[:idx+1])
            next_doc = splits[idx + 1]
            
            curr_chunk_docs_embed = chunker.encoder([curr_chunk_docs])[0]
            next_doc_embed = chunker.encoder([next_doc])[0]
            curr_sim_score = np.dot(curr_chunk_docs_embed, next_doc_embed) / (
                np.linalg.norm(curr_chunk_docs_embed) * np.linalg.norm(next_doc_embed)
            )
            similarities.append(curr_sim_score)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(similarities, marker='o', linestyle='-', color='b')
        ax.axhline(y=chunker.score_threshold, color='r', linestyle='--', 
                  label=f'Threshold: {chunker.score_threshold:.2f}')
        ax.set_title('Similarity Scores Between Cumulative Text and Next Split')
        ax.set_xlabel('Split Index')
        ax.set_ylabel('Similarity Score')
        ax.grid(True)
        ax.legend()
        
        # Add similarity scores as text
        for i, score in enumerate(similarities):
            ax.text(i, score, f'{score:.2f}', ha='center', va='bottom')
        
        return fig, splits, similarities, []
    
    # Step 4: Final chunks
    else:
        splits = chunker._split(text)
        chunks = chunker._chunk(splits)
        
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
    st.title("Chunker Visualization")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choose a chunker:", 
                           ["Statistical Chunker", "Consecutive Chunker", 
                            "Cumulative Chunker"])
    
    # Text input
    text = st.text_area(
        "Enter your text here:",
        height=200,
        help="Enter the text you want to chunk."
    )
    
    if not text:
        st.warning("Please enter some text first!")
        return
    
    if page == "Statistical Chunker":
        st.header("Statistical Chunker")
        
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
        
        # Step navigation
        st.subheader("Chunking Process")
        current_step = st.session_state.get('statistical_step', 1)
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("Previous Step") and current_step > 1:
                current_step -= 1
                st.session_state.statistical_step = current_step
            if st.button("Next Step") and current_step < 6:
                current_step += 1
                st.session_state.statistical_step = current_step
        
        with col2:
            st.progress(current_step / 6)
            st.text(f"Step {current_step}/6")
        
        # Show current step visualization
        with st.spinner("Processing..."):
            fig, splits, similarities, split_indices = visualize_statistical_step(
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
    
    elif page == "Consecutive Chunker":
        st.header("Consecutive Chunker")
        
        # Initialize the encoder and chunker
        with st.spinner("Initializing encoder..."):
            encoder = GigaChatEncoder()
            chunker = ConsecutiveChunker(
                encoder=encoder,
                score_threshold=0.45
            )
        
        # Chunker parameters
        st.subheader("Chunker Parameters")
        score_threshold = st.slider(
            "Score Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.45,
            step=0.01,
            help="Similarity threshold for splitting chunks"
        )
        
        # Update chunker parameters
        chunker.score_threshold = score_threshold
        
        # Step navigation
        st.subheader("Chunking Process")
        current_step = st.session_state.get('consecutive_step', 1)
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("Previous Step") and current_step > 1:
                current_step -= 1
                st.session_state.consecutive_step = current_step
            if st.button("Next Step") and current_step < 4:
                current_step += 1
                st.session_state.consecutive_step = current_step
        
        with col2:
            st.progress(current_step / 4)
            st.text(f"Step {current_step}/4")
        
        # Show current step visualization
        with st.spinner("Processing..."):
            fig, splits, similarities, split_indices = visualize_consecutive_step(
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
            
            elif current_step == 4:
                st.write("Final chunks:")
                chunks = chunker._chunk(splits)
                for i, chunk in enumerate(chunks):
                    with st.expander(f"Chunk {i+1} (Score: {chunk.triggered_score if chunk.triggered_score else 'N/A'})"):
                        st.write(" ".join(chunk.splits))
    
    else:  # Cumulative Chunker
        st.header("Cumulative Chunker")
        
        # Initialize the encoder and chunker
        with st.spinner("Initializing encoder..."):
            encoder = GigaChatEncoder()
            chunker = CumulativeChunker(
                encoder=encoder,
                score_threshold=0.45
            )
        
        # Chunker parameters
        st.subheader("Chunker Parameters")
        score_threshold = st.slider(
            "Score Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.45,
            step=0.01,
            help="Similarity threshold for splitting chunks"
        )
        
        # Update chunker parameters
        chunker.score_threshold = score_threshold
        
        # Step navigation
        st.subheader("Chunking Process")
        current_step = st.session_state.get('cumulative_step', 1)
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("Previous Step") and current_step > 1:
                current_step -= 1
                st.session_state.cumulative_step = current_step
            if st.button("Next Step") and current_step < 4:
                current_step += 1
                st.session_state.cumulative_step = current_step
        
        with col2:
            st.progress(current_step / 4)
            st.text(f"Step {current_step}/4")
        
        # Show current step visualization
        if current_step == 2:
            st.info("На этом шаге происходит подготовка к сравнению: текст разбит на сплиты, далее будет происходить накопление для сравнения.")
        elif current_step == 3:
            splits = chunker._split(text)
            st.subheader("Визуализация накопления текста и сравнения на каждом шаге")
            similarities = []
            curr_chunk_start_idx = 0
            for idx in range(len(splits) - 1):
                if idx == 0:
                    curr_chunk_docs = splits[idx]
                else:
                    curr_chunk_docs = "\n".join(splits[curr_chunk_start_idx : idx + 1])
                next_doc = splits[idx + 1]
                curr_chunk_docs_embed = chunker.encoder([curr_chunk_docs])[0]
                next_doc_embed = chunker.encoder([next_doc])[0]
                curr_sim_score = np.dot(curr_chunk_docs_embed, next_doc_embed) / (
                    np.linalg.norm(curr_chunk_docs_embed) * np.linalg.norm(next_doc_embed)
                )
                similarities.append(curr_sim_score)
                # Визуализация накопления
                st.markdown(f"**Шаг {idx+1}:**")
                html = ""
                if curr_chunk_start_idx < idx:
                    html += f"<span style='color:gray'>{'<br>'.join(splits[curr_chunk_start_idx:idx])}<br></span>"
                html += f"<span style='color:blue'>{splits[idx]}</span>"
                html += "<br><span style='color:red'>Следующий сплит:<br>"
                html += splits[idx+1] + "</span>"
                st.markdown(html, unsafe_allow_html=True)
                st.markdown(f"**Similarity:** {curr_sim_score:.3f}")
                st.markdown("---")
                # Если был разрыв, обновляем curr_chunk_start_idx
                if curr_sim_score < chunker.score_threshold:
                    curr_chunk_start_idx = idx + 1
            st.info("Серым — предыдущий текст в чанке, синим — текущий кумулятивный, красным — следующий сплит для сравнения.")
            # Таблица similarity
            st.write("Similarity scores between cumulative text and next split:")
            df = pd.DataFrame({
                'Step': range(1, len(similarities) + 1),
                'Similarity Score': similarities
            })
            st.dataframe(df)
        else:
            with st.spinner("Processing..."):
                fig, splits, similarities, split_indices = visualize_cumulative_step(
                    text, chunker, current_step
                )
                st.pyplot(fig)
                # Show additional information based on current step
                if current_step == 1:
                    st.write("Initial splits of the text:")
                    for i, split in enumerate(splits):
                        st.text(f"Split {i+1}: {split}")
                elif current_step == 4:
                    st.write("Final chunks:")
                    chunks = chunker._chunk(splits)
                    for i, chunk in enumerate(chunks):
                        with st.expander(f"Chunk {i+1} (Score: {chunk.triggered_score if chunk.triggered_score else 'N/A'})"):
                            st.write(" ".join(chunk.splits))

if __name__ == "__main__":
    main()