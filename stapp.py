import streamlit as st
import matplotlib.pyplot as plt
import torch
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
import requests
from transformers import BertTokenizer, BertForSequenceClassification
import warnings

# Set up the page configuration and suppress warnings
st.set_page_config(layout="wide")
warnings.filterwarnings("ignore", category=FutureWarning)

# Cache the function to get filenames from GitHub
@st.cache
def get_github_files(user, repo, path):
    """Fetches filenames in a specific path from a GitHub repo."""
    url = f"https://api.github.com/repos/{user}/{repo}/contents/{path}"
    response = requests.get(url)
    if response.status_code == 200:
        files = response.json()
        return [file['name'] for file in files if file['name'].endswith('.txt')]
    else:
        return []

# Cache the function to load data from URL
@st.cache
def load_data_from_url(url):
    """Loads text content directly from a provided URL."""
    response = requests.get(url)
    return response.text if response.status_code == 200 else "Error: Unable to retrieve data"

# Cache the model loading for efficiency
@st.cache(allow_output_mutation=True)
def load_model():
    """Load and cache the tokenizer and BERT model."""
    tokenizer = BertTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    model = BertForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    return tokenizer, model

tokenizer, model = load_model()

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
            scores.append(output.logits.argmax(dim=1).item())
    return sentences, scores

def plot_sentiment_scores(scores):
    """Plots a line chart of sentiment scores with a moving average."""
    ma_scores = np.convolve(scores, np.ones(5)/5, mode='valid')
    plt.figure(figsize=(10, 5))
    plt.plot(ma_scores, 'g-', label='Moving Average (window 5)')
    plt.title('Sentiment Analysis Trends')
    plt.xlabel('Sentence Number')
    plt.ylabel('Sentiment Score')
    plt.ylim(0, 4)
    plt.grid(True)
    st.pyplot(plt)

def main():
    st.title('Sentiment Analysis Tool')
    user, repo, path = 'phisles', 'local-sentiment', 'data'
    files = get_github_files(user, repo, path)

    if files:
        selected_file = st.selectbox('Choose a file to analyze:', files)
        file_url = f"https://raw.githubusercontent.com/{user}/{repo}/main/{path}/{selected_file}"
        file_content = load_data_from_url(file_url)

        if st.button('Analyze') and file_content:
            sentences, scores = perform_sentiment_analysis(file_content)
            st.write("Annotated Sentences:")  # Display the header

            col1, col2 = st.columns([2, 3])  # Set up columns for text and graph
            full_text = ""  # Initialize text accumulation
            all_scores = []  # List to store scores for plotting

            for sentence, score in zip(sentences, scores):
                full_text += f"<span style='background-color:{determine_color(score)};'>{sentence}</span> "
                all_scores.append(score)
                with col1:
                    st.markdown(full_text, unsafe_allow_html=True)
                with col2:
                    plot_sentiment_scores(all_scores)

                import time
                time.sleep(0.2)  # Add delay to simulate streaming

def determine_color(score):
    """Determines color based on sentiment score."""
    return ("rgba(255, 77, 77, 0.6)" if score == 0 else
            "rgba(255, 182, 193, 0.6)" if score == 1 else
            "" if score == 2 else
            "rgba(144, 238, 144, 0.6)" if score == 3 else
            "rgba(0, 100, 0, 0.6)" if score == 4 else
            "rgba(77, 77, 255, 0.6)")

if __name__ == "__main__":
    main()
