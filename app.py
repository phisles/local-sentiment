import os
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import warnings

# Suppress specific FutureWarnings from huggingface_hub
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")

print("TRANSCRIPT SENTIMENT ANALYSIS - SCORE OF 0-4 (0 = NEGATIVE, 4 = POSITIVE)")

# Initialize the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = BertForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

def tokenize_and_classify(content, max_length):
    start = 0
    while start < len(content):
        # Take a substring of the content
        substring = content[start:start+max_length]
        tokens = tokenizer.encode(substring, add_special_tokens=True)
        if len(tokens) > 512:
            # Reduce the substring length if the token count is exceeded
            while len(tokens) > 512:
                substring = substring[:-100]  # Trim substring
                tokens = tokenizer.encode(substring, add_special_tokens=True)
        yield substring, tokens
        start += len(substring)

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
            for text_snippet, tokens in tokenize_and_classify(content, 1000):  # Character-based chunk size
                label = classify_tokens(tokens)
                print(f"Sentiment Score: {label}")
                print(f"Text Snippet: {text_snippet[:60]}...")  # Print the first 50 characters of the snippet and a line break
            print("_" * 50)  # Prints a line of underscores after each file's output
            print("\n")
#test
