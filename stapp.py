# sentiment_streamlit_app.py
import streamlit as st
import os
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertForSequenceClassification, DistilBertTokenizer, DistilBertForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification, XLNetTokenizer, XLNetForSequenceClassification, ElectraTokenizer, ElectraForSequenceClassification, GPT2Tokenizer, GPT2LMHeadModel
import torch
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from annotated_text import annotated_text
import warnings
from streamlit_extras.streaming_write import write

# Suppress specific FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load NLTK's Punkt tokenizer for sentence splitting if you havent already
#nltk.download('punkt', quiet=True)

tokenizer = BertTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = BertForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')


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

def display_annotated_text(sentences, scores):
    annotated_results = []
    for sentence, score in zip(sentences, scores):
        if score == 0:
            color = "rgba(255, 77, 77, 0.6)"  # Red for the most negative sentiment
        elif score == 1:
            color = "rgba(255, 182, 193, 0.6)"  # Pink for slightly negative sentiment
        elif score == 2:
            color = ""  # No background color for neutral sentiment
        elif score == 3:
            color = "rgba(144, 238, 144, 0.6)"  # Light Green for positive sentiment
        elif score == 4:
            color = "rgba(0, 100, 0, 0.6)"  # Dark Green for the most positive sentiment
        else:
            color = "rgba(77, 77, 255, 0.6)"  # Default blue for any scores out of expected range
        annotated_results.append((sentence, "", color))
    annotated_text(*annotated_results)


def main():
    st.title('Sentiment Analysis Tool')
    data_path = '/Users/philip/Desktop/Code/Sentiment/data'
    files_content = load_data(data_path)

    if files_content:
        selected_file = st.selectbox('Choose a file to analyze:', list(files_content.keys()))
        if st.button('Analyze'):
            sentences, scores = perform_sentiment_analysis(files_content[selected_file])
            plot_sentiment_scores(scores)  # Display the line chart with moving average
            st.write("Annotated Sentences:")  # Separator for annotated text
            display_annotated_text(sentences, scores)  # Display the annotated text
    else:
        st.write("No text files found in the directory.")

if __name__ == "__main__":
    main()
