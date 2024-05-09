import os
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import nltk
import warnings

# Suppress specific FutureWarnings from huggingface_hub
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")

# Initialize the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = BertForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

# Load NLTK's Punkt tokenizer for sentence splitting
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

def tokenize_and_classify(content):
    # Split the content into sentences using NLTK
    sentences = sent_tokenize(content)
    for sentence in sentences:
        tokens = tokenizer.encode(sentence, add_special_tokens=True)
        # Check if the tokens exceed the maximum allowed, and skip if they do
        if len(tokens) <= 512:
            yield sentence, tokens

def classify_tokens(tokens):
    input_ids = torch.tensor([tokens]).to(model.device)
    with torch.no_grad():
        output = model(input_ids)
    probabilities = torch.softmax(output.logits, dim=1)
    return probabilities.argmax(dim=1).item()

# Directory path to the text files
directory_path = '/Users/philip/Desktop/Code/Sentiment/data'

# Loop through each text file in the directory
for filename in os.listdir(directory_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(directory_path, filename)
        with open(file_path, 'r') as file:
            content = file.read()
            print(f"----File: {filename}----")
            for text_snippet, tokens in tokenize_and_classify(content):
                label = classify_tokens(tokens)
                print(f"Sentiment Score: {label}")
                print(f"Text Snippet: {text_snippet}\n")
            print("_" * 50)  # Prints a line of underscores after each file's output
            print("\n")
