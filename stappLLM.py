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
import transformers
import torch 


st.set_page_config(layout="wide")

# Suppress specific FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load NLTK's Punkt tokenizer for sentence splitting if you havent already
nltk.download('punkt', quiet=True)

@st.cache_resource
def load_model():
    """Load and cache the tokenizer and BERT model."""
    tokenizer = BertTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    model = BertForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    return tokenizer, model

@st.cache_resource
def load_llama_model():
    """Load and cache the LLaMA model from Hugging Face using the API token from Streamlit secrets."""
    model_id = "meta-llama/Meta-Llama-3-8B"
    # Retrieve the Hugging Face API token securely stored in Streamlit Cloud secrets
    hf_token = st.secrets["huggingface"]["token"]

    # Initialize the pipeline with the API token for authentication
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},  # Using reduced precision to save memory
        device_map="auto",  # Automatically use the best available device (GPU or CPU)
        use_auth_token=hf_token  # Authenticate with the API token
    )
    return pipeline

# Load the model using the secure token at the start of your script
llama_pipeline = load_llama_model()


@st.cache_data
def load_data(files):
    """Loads text files uploaded by the user and returns a dictionary of filename and content."""
    files_content = {}
    for file_name, file_content in files.items():
        files_content[file_name] = file_content.decode("utf-8")
    return files_content

def perform_sentiment_analysis(text, tokenizer, model):
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

def generate_feedback(sentences, scores, pipeline):
    """Generate feedback using the LLaMA model based on the provided sentences and sentiment scores."""
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

    # Sending the prompt to the LLaMA model
    response = pipeline(prompt_text, max_length=512)  # You can adjust max_length as needed

    # Extracting the generated text from the response
    # The output from Hugging Face's pipeline is a list of generated texts, so we pick the first.
    generated_text = response[0]['generated_text']
    return generated_text


def main():
    st.title('Sentiment Analysis Tool')
    tokenizer, model = load_model()  # Load the tokenizer and model only once at the start

    uploaded_files = st.file_uploader("Upload Files", type=['txt'], accept_multiple_files=True)
    if uploaded_files:
        file_dict = {file.name: file for file in uploaded_files}  # Create a dictionary from uploaded files
        selected_file = st.selectbox('Choose a file to analyze:', list(file_dict.keys()))

        if st.button('Analyze'):
            file_to_analyze = file_dict[selected_file]
            file_content = file_to_analyze.getvalue().decode("utf-8")  # Decode the file content
            sentences, scores = perform_sentiment_analysis(file_content, tokenizer, model)  # Pass tokenizer and model

            col1, col2, col3 = st.columns([2, 3, 2])  # Adjust column width ratios as needed

            text_container = col1.empty()  # Container for streaming text
            chart_container = col2.empty()  # Container for the line chart
            feedback_container = col3.empty()  # Container for feedback

            full_text = ""  # Initialize empty string to accumulate text
            all_scores = []  # List to store scores for plotting

            for sentence, score in zip(sentences, scores):
                color = determine_color(score)
                full_text += f"<span style='background-color:{color};'>{sentence}</span> "
                all_scores.append(score)
                text_container.markdown(full_text, unsafe_allow_html=True)
                update_line_chart(chart_container, all_scores)

            llama_pipeline = load_llama_model()
            feedback = generate_feedback(sentences, scores, llama_pipeline)
            feedback = feedback.replace('•', '-').replace('\n\n', '\n')  # Replace bullets and manage extra new lines

            # Add extra line breaks before each bold section except the first
            parts = feedback.split('**')
            formatted_feedback = parts[0]  # Start with the first part that is before the first bold
            for part in parts[1:]:
                if formatted_feedback.count('**') % 2 == 0:  # Check if we're at the start of bold text
                    formatted_feedback += '\n**' + part
                else:
                    formatted_feedback += '**' + part

            feedback_container.markdown(formatted_feedback, unsafe_allow_html=True)  # Stream feedback directly







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