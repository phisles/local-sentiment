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
import ollama
import time


st.set_page_config(layout="wide")

# Suppress specific FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)
# Suppress Matplotlib warnings about too many open figures
plt.rcParams['figure.max_open_warning'] = 100  # You can set this to a higher value


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

    #uncomment to view sentiment score output:
    #print(scores)
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
    plt.close()  # Close the figure to avoid memory issues

def generate_feedback(sentences, scores):
    # Prepare the combined text
    combined_text = ". ".join([f"{sentence} [Score: {score}]" for sentence, score in zip(sentences, scores)])
    prompt_text = f"""
    You are an analyst reviewing transcriptions of body worn cameras and interviews along with sentiment scores for each sentence.
    Format your response with:
    - Bullet points for each sentiment analysis observation
    - Separate paragraphs for summary, sentiment analysis, feedback for officials, and additional insights
    - Ensure to use concise language

    Here are the sentences and their sentiment scores:
    {combined_text}
    """

    # Sending the prompt to LLaMA
    response = ollama.chat(
        model='llama3',
        messages=[{'role': 'user', 'content': prompt_text}]
    )

    # Extracting the content from the response
    return response['message']['content']


def main():
    st.title('Sentiment Analysis Tool')
    data_path = '/Users/philip/Desktop/Code/Sentiment/data'
    files_content = load_data(data_path)

    if files_content:
        selected_file = st.selectbox('Choose a file to analyze:', list(files_content.keys()))
        if st.button('Analyze'):
            sentences, scores = perform_sentiment_analysis(files_content[selected_file])
            col1, col2, col3 = st.columns([2, 3, 2])  # Adjust column width ratios as needed

            text_container = col1.empty()  # Container for streaming text
            chart_container = col2.empty()  # Container for the line chart
            feedback_container = col3.empty()  # Container for feedback

            full_text = ""  # Initialize empty string to accumulate text
            all_scores = []  # List to store scores for plotting

            for i, (sentence, score) in enumerate(zip(sentences, scores)):
                color = determine_color(score)
                full_text += f"<span style='background-color:{color};'>{sentence}</span> "
                all_scores.append(score)

                text_container.markdown(full_text, unsafe_allow_html=True)
                update_line_chart(chart_container, all_scores)

                # Generate feedback after processing all sentences
            feedback = generate_feedback(sentences, all_scores)

            # Container for feedback
            feedback_container = col3.empty()

            # Convert special bullet points to Markdown bullets and ensure proper line breaks
            feedback = feedback.replace('•', '-').replace('\n\n', '\n')  # Replace bullets and manage extra new lines

            # Add extra line breaks before each bold section except the first
            parts = feedback.split('**')
            formatted_feedback = parts[0]  # Start with the first part that is before the first bold
            for part in parts[1:]:  # Loop through parts after the first bold
                if formatted_feedback.count('**') % 2 == 0:  # Check if we're at the start of bold text
                    formatted_feedback += '\n**' + part
                else:
                    formatted_feedback += '**' + part

            # Stream feedback directly (considering it's properly formatted now)
            feedback_container.markdown(formatted_feedback, unsafe_allow_html=True)


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