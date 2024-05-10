import streamlit as st
import os
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
import warnings
import requests
import json
import time

st.set_page_config(layout="wide")
warnings.filterwarnings("ignore", category=FutureWarning)

tokenizer = BertTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = BertForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

@st.cache_data
def load_data(directory):
    files_content = {}
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as file:
                files_content[filename] = file.read()
    return files_content

def perform_sentiment_analysis(text):
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

def determine_color(score):
    return ("rgba(255, 77, 77, 0.6)" if score == 0 else
            "rgba(255, 182, 193, 0.6)" if score == 1 else
            "" if score == 2 else
            "rgba(144, 238, 144, 0.6)" if score == 3 else
            "rgba(0, 100, 0, 0.6)" if score == 4 else
            "rgba(77, 77, 255, 0.6)")

def stream_annotated_text(sentences, scores):
    for sentence, score in zip(sentences, scores):
        color = determine_color(score)
        html_text = f"<span style='background-color:{color};'>{sentence}</span> "
        yield html_text, score

def main():
    st.title('Sentiment Analysis Tool')
    data_path = '/Users/philip/Desktop/Code/Sentiment/data'
    files_content = load_data(data_path)

    if files_content:
        selected_file = st.selectbox('Choose a file to analyze:', list(files_content.keys()))
        if st.button('Analyze'):
            sentences, scores = perform_sentiment_analysis(files_content[selected_file])
            col1, col2, col3 = st.columns([2, 3, 2])  # Create three columns

            text_stream = stream_annotated_text(sentences, scores)
            text_container = col1.empty()
            chart_container = col2.empty()
            feedback_container = col3.empty()
            feedback_container.write("Feedback and Suggestions:")

            full_text = ""
            all_scores = []

            for html_text, score in text_stream:
                full_text += html_text
                all_scores.append(score)

                text_container.markdown(full_text, unsafe_allow_html=True)
                
                if len(all_scores) >= 5:  # Update chart if we have enough data to start moving average
                    ma_scores = np.convolve(all_scores, np.ones(5)/5, mode='valid')
                    plt.figure(figsize=(10, 5))
                    plt.plot(ma_scores, 'g-', label='Moving Average (window 5)')
                    plt.title('Sentiment Analysis Trends')
                    plt.xlabel('Sentence Number')
                    plt.ylabel('Sentiment Score')
                    plt.ylim(0, 4)
                    plt.grid(True)
                    chart_container.pyplot(plt)

                feedback_html_text = f"<span style='background-color:{determine_color(2)};'>Feedback coming here...</span> "
                feedback_container.markdown(feedback_html_text, unsafe_allow_html=True)  # Example of streaming feedback

                time.sleep(0.1)  # Simulate delay for streaming effect

if __name__ == "__main__":
    main()
