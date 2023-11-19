import random
from nltk import ngrams
from collections import Counter
import re
import math

# Load the corpus
with open('GPAC.txt', 'r', encoding='utf-8') as file:
    corpus = file.read()

# Preprocess the corpus (remove punctuation, convert to lowercase)
corpus = re.sub(r'[^\w\s]', '', corpus)
corpus = corpus.lower()

# Tokenize the corpus
tokens = corpus.split()

# Function to generate n-grams
def generate_ngrams(tokens, n):
    return list(ngrams(tokens, n))

# Create n-grams for n=1, 2, 3, 4
ngram_1 = generate_ngrams(tokens, 1)
ngram_2 = generate_ngrams(tokens, 2)
ngram_3 = generate_ngrams(tokens, 3)
ngram_4 = generate_ngrams(tokens, 4)

# Function to calculate n-gram probabilities
def calculate_probabilities(ngram_list):
    ngram_counts = Counter(ngram_list)
    total_ngrams = len(ngram_list)
    probabilities = {ngram: count / total_ngrams for ngram, count in ngram_counts.items()}
    return probabilities

# Calculate probabilities for each n-gram
prob_1 = calculate_probabilities(ngram_1)
prob_2 = calculate_probabilities(ngram_2)
prob_3 = calculate_probabilities(ngram_3)
prob_4 = calculate_probabilities(ngram_4)

# Function to calculate perplexity
def calculate_perplexity(ngram_probabilities, n, test_set):
    test_set_ngrams = generate_ngrams(test_set, n)
    total_ngrams = len(test_set_ngrams)
    log_prob_sum = 0
    for ngram in test_set_ngrams:
        log_prob = math.log(ngram_probabilities.get(ngram, 1e-10))  # Use a small value for unknown ngrams
        log_prob_sum += log_prob
    avg_log_prob = log_prob_sum / total_ngrams
    perplexity = math.exp(-avg_log_prob)
    return perplexity

# Generate a test set
test_set = random.choices(tokens, k=1000)  # Adjust the size of the test set as needed

# Calculate perplexity for each n-gram model
perplexity_1 = calculate_perplexity(prob_1, 1, test_set)
perplexity_2 = calculate_perplexity(prob_2, 2, test_set)
perplexity_3 = calculate_perplexity(prob_3, 3, test_set)
perplexity_4 = calculate_perplexity(prob_4, 4, test_set)

# Show the perplexity values
print("Perplexity for N-gram 1:", perplexity_1)
print("Perplexity for N-gram 2:", perplexity_2)
print("Perplexity for N-gram 3:", perplexity_3)
print("Perplexity for N-gram 4:", perplexity_4)