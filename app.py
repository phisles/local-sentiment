import os
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import nltk
import warnings

# Suppress specific FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Initialize the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = BertForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

# Download and load the NLTK sentence tokenizer
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

def process_content(content, max_tokens):
    sentences = sent_tokenize(content)
    grouped_text = ""
    first_sentence = None
    last_sentence = ""
    for sentence in sentences:
        # Check if adding this sentence stays within the token limit
        new_group = grouped_text + " " + sentence if grouped_text else sentence
        if len(tokenizer.encode(new_group, add_special_tokens=True)) > max_tokens:
            # If adding exceeds the limit, process the current group if it exists
            if grouped_text:
                yield first_sentence, last_sentence, grouped_text
            grouped_text = sentence  # Start a new group with the current sentence
            first_sentence = sentence
            last_sentence = sentence
        else:
            # If within limit, add to current group
            grouped_text = new_group
            last_sentence = sentence
    # Yield the last group if it exists
    if grouped_text:
        yield first_sentence, last_sentence, grouped_text

def classify_tokens(text):
    tokens = tokenizer.encode(text, add_special_tokens=True)
    input_ids = torch.tensor([tokens]).to(model.device)
    with torch.no_grad():
        output = model(input_ids)
    return output.logits.argmax(dim=1).item()

# Directory path to the text files
directory_path = '/Users/philip/Desktop/Code/Sentiment/data'

# Loop through each text file in the directory
for filename in os.listdir(directory_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(directory_path, filename)
        with open(file_path, 'r') as file:
            content = file.read()
            print(f"----File: {filename}----")
            for first, last, group in process_content(content, 500):
                sentiment_score = classify_tokens(group)
                print(f"Sentiment Score: {sentiment_score}")
                print(f"First Sentence: {first}")
                print(f"Last Sentence: {last}\n")
            print("_" * 50 + "\n")
