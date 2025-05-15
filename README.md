[![Aurelio AI](https://pbs.twimg.com/profile_banners/1671498317455581184/1696285195/1500x500)](https://aurelio.ai)

# Semantic Chunkers

<p>
<img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/semantic-chunkers?logo=python&logoColor=gold" />
<img alt="GitHub Contributors" src="https://img.shields.io/github/contributors/aurelio-labs/semantic-chunkers" />
<img alt="GitHub Last Commit" src="https://img.shields.io/github/last-commit/aurelio-labs/semantic-chunkers" />
<img alt="" src="https://img.shields.io/github/repo-size/aurelio-labs/semantic-chunkers" />
<img alt="GitHub Issues" src="https://img.shields.io/github/issues/aurelio-labs/semantic-chunkers" />
<img alt="GitHub Pull Requests" src="https://img.shields.io/github/issues-pr/aurelio-labs/semantic-chunkers" />
<img src="https://codecov.io/gh/aurelio-labs/semantic-chunkers/graph/badge.svg?token=H8OOMV2TUF" />
<img alt="Github License" src="https://img.shields.io/badge/License-MIT-yellow.svg" />
</p>

Semantic Chunkers is a multi-modal chunking library for intelligent chunking of text, video, and audio. It makes your AI and data processing more efficient _and_ accurate.

---

## 📚 Resources

### Docs

| Notebook | Description |
| -------- | ----------- |
| [Introduction](https://github.com/aurelio-labs/semantic-chunkers/blob/main/docs/01-video-chunking.ipynb) | Chunking videos with semantics |

# StatisticalChunker Visualization

This application provides a visual interface for understanding how the StatisticalChunker works. It allows you to input text and see how it gets split into sentences and then chunked based on semantic similarity.

## Features

- Interactive text input
- Adjustable chunking parameters
- Visualization of similarity scores between sentences
- Visualization of chunk sizes
- Detailed statistics about the chunking process
- Expandable view of each generated chunk

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Set your OpenAI API key as an environment variable:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

2. Run the Streamlit application:
```bash
streamlit run app.py
```

3. Open your web browser and navigate to the URL shown in the terminal (usually http://localhost:8501)

4. Enter your text in the text area and adjust the chunking parameters if needed

5. Click "Create Chunks" to see the visualization and results

## Parameters

- **Window Size**: Number of previous sentences to consider when calculating similarity scores
- **Minimum Tokens per Chunk**: Minimum number of tokens that should be in each chunk

## Visualization

The application shows two main visualizations:

1. **Similarity Scores**: A line plot showing the similarity scores between consecutive sentences
2. **Chunk Sizes**: A bar plot showing how many sentences are in each chunk

Additionally, you can expand each chunk to see its contents and the similarity score that triggered its creation.


