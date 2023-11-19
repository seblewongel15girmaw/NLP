import random
from nltk import ngrams
from collections import Counter
import re

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

# Function to generate a random sentence using n-grams
def generate_random_sentence(ngram_probabilities, n):
    sentence = []
    for _ in range(n):
        candidate_ngrams = [ngram for ngram in ngram_probabilities.keys() if len(ngram) == n]
        next_ngram = random.choices(
            candidate_ngrams,
            weights=[ngram_probabilities[ngram] for ngram in candidate_ngrams]
        )[0]
        sentence.extend(list(next_ngram))
    return sentence

# Generate random sentences for different n values
random_sentence_1 = generate_random_sentence(prob_1, 1)
random_sentence_2 = generate_random_sentence(prob_2, 2)
random_sentence_3 = generate_random_sentence(prob_3, 3)
random_sentence_4 = generate_random_sentence(prob_4, 4)

# Convert the random sentences to strings
random_sentence_1 = ' '.join(random_sentence_1)
random_sentence_2 = ' '.join(random_sentence_2)
random_sentence_3 = ' '.join(random_sentence_3)
random_sentence_4 = ' '.join(random_sentence_4)

# Show the generated sentences
print("Random sentence for N-gram 1:", random_sentence_1)
print("Random sentence for N-gram 2:", random_sentence_2)
print("Random sentence for N-gram 3:", random_sentence_3)
print("Random sentence for N-gram 4:", random_sentence_4)


# As n increases, the generated sentences tend to have better grammatical structure, contextual coherence, and semantic meaning