from nltk import ngrams
from collections import Counter
import re

# Load the corpus
with open('GPAC.txt', 'r', encoding='utf-8') as file:
    corpus = file.read()

# Preprocess the corpus (remove punctuation, convert to lowercase)
corpus = re.sub(r'[^\w\s]', '', corpus)
corpus = corpus.lower()


tokens = corpus.split()

# Function to generate n-grams
def generate_ngrams(tokens, n):
    return list(ngrams(tokens, n))

# Function to calculate n-gram probabilities
def calculate_probabilities(ngram_list):
    ngram_counts = Counter(ngram_list)
    total_ngrams = len(ngram_list)
    probabilities = {ngram: count / total_ngrams for ngram, count in ngram_counts.items()}
    return probabilities


def sentence_probability(sentence, n):
    sentence_tokens = sentence.split()
    ngram_list = generate_ngrams(tokens, n)
    ngram_probabilities = calculate_probabilities(ngram_list)
    sentence_ngrams = generate_ngrams(sentence_tokens, n)
    prob = 1.0
    for ngram in sentence_ngrams:
        prob *= ngram_probabilities.get(ngram, 0.0)  # Use Laplace smoothing
    return prob


prob_sentence_1 = sentence_probability("ኢትዮጵያ ታሪካዊ ሀገር ናት", 1)
prob_sentence_2 = sentence_probability("ኢትዮጵያ ታሪካዊ ሀገር ናት", 2)
prob_sentence_3 = sentence_probability("ኢትዮጵያ ታሪካዊ ሀገር ናት", 3)
prob_sentence_4 = sentence_probability("ኢትዮጵያ ታሪካዊ ሀገር ናት", 4)

# Show the probabilities
print("Probability of the sentence 'ኢትዮጵያ ታሪካዊ ሀገር ናት' for all n-grams:")
print("Probability for N-gram 1:", prob_sentence_1)
print("Probability for N-gram 2:", prob_sentence_2)
print("Probability for N-gram 3:", prob_sentence_3)
print("Probability for N-gram 4:", prob_sentence_4)