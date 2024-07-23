# Sentiment Streamlit App

## Overview

This Streamlit application performs sentiment analysis on text files. It uses BERT for sequence classification to analyze sentiments in sentences, plots sentiment score trends, and provides feedback on the analysis. The app leverages various tools such as Streamlit, Matplotlib, and Ollama for generating feedback based on sentiment scores.

## Features

- **Sentiment Analysis**: Analyze sentiment for each sentence using BERT.
- **Visualization**: Plot sentiment scores with a moving average trend.
- **Feedback Generation**: Generate concise feedback based on sentiment scores using Ollama's LLaMA.

## Prerequisites

- Python 3.x
- Required libraries (install via `pip`):
  - `streamlit`
  - `matplotlib`
  - `transformers`
  - `torch`
  - `numpy`
  - `nltk`
  - `annotated_text`
  - `ollama`
  - `streamlit_extras`

## Installation

1. **Clone the Repository** (if applicable):
    ```bash
    git clone <repository_url>
    ```

2. **Install Required Packages**:
    ```bash
    pip install streamlit matplotlib transformers torch numpy nltk annotated_text ollama streamlit_extras
    ```

3. **Download NLTK Data**:
    Make sure NLTK's Punkt tokenizer is downloaded:
    ```python
    import nltk
    nltk.download('punkt', quiet=True)
    ```

## Usage

1. **Prepare Your Data**:
    - Place your text files in the directory specified in the script (`data_path`).

2. **Run the Application**:
    ```bash
    streamlit run sentiment_streamlit_app.py
    ```

3. **Interact with the App**:
    - Select a text file from the dropdown menu.
    - Click the "Analyze" button to perform sentiment analysis.
    - View the annotated text, sentiment score chart, and feedback.


## Functions

load_data(directory): Loads text files from the specified directory.
perform_sentiment_analysis(text): Analyzes sentiment of the provided text and returns sentences and scores.
plot_sentiment_scores(scores): Plots sentiment scores with a moving average trend.
generate_feedback(sentences, scores): Generates feedback using LLaMA based on sentiment scores.
stream_annotated_text(sentences, scores): Generates HTML styled text for streaming.
determine_color(score): Determines color based on sentiment score.
update_line_chart(container, scores): Updates the line chart with moving average scores.
Notes
Ensure the required models and libraries are properly set up.
This application uses BertTokenizer and BertForSequenceClassification for sentiment analysis.
The feedback generation part requires integration with Ollama's LLaMA model.
