from nltk import ngrams
from collections import Counter
import re
import math

def read_corpus(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        corpus = file.read()
    return corpus

def preprocess_corpus(corpus):
    corpus = re.sub(r'[^\w\s]', '', corpus)
    return corpus.lower()

def tokenize_corpus(corpus):
    return corpus.split()

def generate_ngrams(tokens, n):
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngrams.append(tuple(tokens[i:i+n]))
    return ngrams

# a function that calculates the perplexity of a given n-gram model on a sequence of test_tokens
def calculate_perplexity(ngram_model, test_tokens):
    total_log_probability = 0
    total_tokens = len(test_tokens)

    for i in range(len(test_tokens) - len(ngram_model) + 1):
        context = tuple(test_tokens[i:i+len(ngram_model)-1])
        token = test_tokens[i+len(ngram_model)-1]
        context_count = ngram_model[context] if context in ngram_model else 0
        token_count = ngram_model[context + (token,)] if context + (token,) in ngram_model else 1
        probability = token_count / (context_count + len(ngram_model))
        total_log_probability += math.log2(probability)

    perplexity = 2 ** (-total_log_probability / total_tokens)
    return perplexity

def main(file_path):
    # Load the corpus
    corpus = read_corpus(file_path)

    # Preprocess the corpus
    corpus = preprocess_corpus(corpus)

    # Tokenize the corpus
    tokens = tokenize_corpus(corpus)

    # Create n-grams for n=1, 2, 3, 4
    ngram_1 = Counter(generate_ngrams(tokens, 1))
    ngram_2 = Counter(generate_ngrams(tokens, 2))
    ngram_3 = Counter(generate_ngrams(tokens, 3))
    ngram_4 = Counter(generate_ngrams(tokens, 4))

    # Load the test corpus
    test_corpus = read_corpus(file_path)
    test_tokens = tokenize_corpus(preprocess_corpus(test_corpus))

    
    perplexity_1 = calculate_perplexity(ngram_1, test_tokens)
    perplexity_2 = calculate_perplexity(ngram_2, test_tokens)
    perplexity_3 = calculate_perplexity(ngram_3, test_tokens)
    perplexity_4 = calculate_perplexity(ngram_4, test_tokens)

   
    print("Intrinsic Perplexity for n-gram models:")
    print("N-gram 1:", perplexity_1)
    print("N-gram 2:", perplexity_2)
    print("N-gram 3:", perplexity_3)
    print("N-gram 4:", perplexity_4)

file_path = 'GPAC.txt'

# Call the main function
main(file_path)