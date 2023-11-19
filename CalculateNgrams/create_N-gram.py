from nltk import ngrams
from collections import Counter
import re

# A function that read corpus file
def read_corpus(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        corpus = file.read()
    return corpus

# function removes non-alphanumeric and non-whitespace characters from the corpus and converts it to lowercase. 
def preprocess_corpus(corpus):
    corpus = re.sub(r'[^\w\s]', '', corpus)
    return corpus.lower()

#  a function splits the corpus into individual tokens based on whitespace characters, providing a list of tokens as the output.
def tokenize_corpus(corpus):
    return corpus.split()

#  a function that generate ngrams
def generate_ngrams(tokens, n):
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngrams.append(tuple(tokens[i:i+n]))
    return ngrams

def main(file_path):
    corpus = read_corpus(file_path)
    corpus = preprocess_corpus(corpus)
    tokens = tokenize_corpus(corpus)

    # Create n-grams for n=1, 2, 3, 4
    ngram_1 = generate_ngrams(tokens, 1)
    ngram_2 = generate_ngrams(tokens, 2)
    ngram_3 = generate_ngrams(tokens, 3)
    ngram_4 = generate_ngrams(tokens, 4)


    print("N-grams Sample OutPut:")
    print("N-gram 1:", ngram_1[:5])
    print("N-gram 2:", ngram_2[:5])
    print("N-gram 3:", ngram_3[:5])
    print("N-gram 4:", ngram_4[:5])

file_path = 'GPAC.txt'

# Call the main function
main(file_path)
