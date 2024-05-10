# sentiment_streamlit_app.py
import streamlit as st
import os
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from annotated_text import annotated_text
import warnings
from streamlit_extras.streaming_write import write


st.set_page_config(layout="wide")

# Suppress specific FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load NLTK's Punkt tokenizer for sentence splitting if you havent already
#nltk.download('punkt', quiet=True)

@st.cache(allow_output_mutation=True)
def load_model():
    """Load and cache the tokenizer and BERT model."""
    tokenizer = BertTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    model = BertForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    return tokenizer, model

tokenizer, model = load_model()


@st.cache_data
def load_data(directory):
    """Loads text files from a directory and returns a dictionary of filename and content."""
    files_content = {}
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as file:
                files_content[filename] = file.read()
    return files_content

def perform_sentiment_analysis(text):
    """Performs sentiment analysis on the provided text and returns a list of sentiment scores."""
    sentences = sent_tokenize(text)
    scores = []
    for sentence in sentences:
        tokens = tokenizer.encode(sentence, add_special_tokens=True)
        if len(tokens) <= 512:
            input_ids = torch.tensor([tokens]).to('cpu')
            with torch.no_grad():
                output = model(input_ids)
            score = output.logits.argmax(dim=1).item()
            scores.append(score)
    return sentences, scores

def plot_sentiment_scores(scores):
    """Plots a line chart of sentiment scores with a moving average."""
    ma_scores = np.convolve(scores, np.ones(5)/5, mode='valid')  # Moving average
    plt.figure(figsize=(10, 5))
    plt.plot(ma_scores, 'g-', label='Moving Average (window 5)')
    plt.title('Sentiment Analysis Trends')
    plt.xlabel('Sentence Number')
    plt.ylabel('Sentiment Score')
    plt.ylim(0, 4)  # Set y-axis to show the full range of scores
    plt.grid(True)
    st.pyplot(plt)




def main():
    st.title('Sentiment Analysis Tool')
    data_path = 'https://github.com/phisles/local-sentiment/blob/main/data/Body%20Cam%203.txt'
    files_content = load_data(data_path)

    if files_content:
        selected_file = st.selectbox('Choose a file to analyze:', list(files_content.keys()))
        if st.button('Analyze'):
            sentences, scores = perform_sentiment_analysis(files_content[selected_file])
            st.write("Annotated Sentences:")  # Header before displaying annotated text

            # Prepare the containers for text and the line chart
            col1, col2 = st.columns([2, 3])  # Adjust column width ratios as needed

            full_text = ""  # Initialize empty string to accumulate text
            all_scores = []  # List to store scores for plotting

            # Streaming text and plotting in columns
            with col1:
                text_container = st.empty()  # Container for streaming text
            with col2:
                chart_container = st.empty()  # Container for the line chart

            # Iterate through text and scores, updating UI elements
            for sentence, score in zip(sentences, scores):
                color = determine_color(score)
                html_text = f"<span style='background-color:{color};'>{sentence}</span> "
                full_text += html_text
                all_scores.append(score)
                
                with col1:
                    text_container.markdown(full_text, unsafe_allow_html=True)
                with col2:
                    update_line_chart(chart_container, all_scores)

                import time
                time.sleep(0.2)  # Simulate delay for streaming effect


def stream_annotated_text(sentences, scores):
    """Generator function to create HTML styled text and scores for streaming."""
    for sentence, score in zip(sentences, scores):
        color = determine_color(score)
        html_text = f"<span style='background-color:{color};'>{sentence}</span> "
        yield html_text, score  # Yield both text and score for use in the main loop

def determine_color(score):
    """Determine color based on score."""
    return ("rgba(255, 77, 77, 0.6)" if score == 0 else
            "rgba(255, 182, 193, 0.6)" if score == 1 else
            "" if score == 2 else
            "rgba(144, 238, 144, 0.6)" if score == 3 else
            "rgba(0, 100, 0, 0.6)" if score == 4 else
            "rgba(77, 77, 255, 0.6)")

def update_line_chart(container, scores):
    """Update the line chart with the moving average of sentiment scores."""
    ma_scores = np.convolve(scores, np.ones(5)/5, mode='valid')  # Calculate moving average
    plt.figure(figsize=(10, 5))
    plt.plot(ma_scores, 'g-', label='Moving Average (window 5)')
    plt.title('Sentiment Analysis Trends')
    plt.xlabel('Sentence Number')
    plt.ylabel('Sentiment Score')
    plt.ylim(0, 4)  # Set y-axis to show the full range of scores
    plt.grid(True)
    container.pyplot(plt)

if __name__ == "__main__":
    main()




