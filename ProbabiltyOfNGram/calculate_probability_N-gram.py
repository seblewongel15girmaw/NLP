from nltk import ngrams
from collections import Counter
import re

def calculate_probabilities(ngram_list):
    ngram_counts = Counter(ngram_list)
    total_ngrams = len(ngram_list)
    probabilities = {ngram: count / total_ngrams for ngram, count in ngram_counts.items()}
    return probabilities

def top_10_ngrams(probabilities):
    return dict(sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:10])

def process_corpus(file_path):
    # Load the corpus
    with open(file_path, 'r', encoding='utf-8') as file:
        corpus = file.read()

    # Preprocess the corpus (remove punctuation, convert to lowercase)
    corpus = re.sub(r'[^\w\s]', '', corpus)
    corpus = corpus.lower()

    # Tokenize the corpus
    tokens = corpus.split()

    # Create a dictionary to store probabilities for each n
    ngram_probabilities = {}

    # Iterate over different values of n
    for n in range(1, 5):
        # Generate n-grams
        ngrams_list = list(ngrams(tokens, n))

        # Calculate probabilities for n-grams
        probabilities = calculate_probabilities(ngrams_list)

        # Find top 10 most likely n-grams
        top_10 = top_10_ngrams(probabilities)

        # Store the probabilities and top 10 n-grams for the current n
        ngram_probabilities[n] = (probabilities, top_10)

    # Show top 10 most likely n-grams for all n
    print("Top 10 most likely n-grams:")
    for n, (probabilities, top_10) in ngram_probabilities.items():
        print("Top 10 N-gram", n, ":", top_10)

# Call the function with the file path
file_path = 'GPAC.txt'
process_corpus(file_path)