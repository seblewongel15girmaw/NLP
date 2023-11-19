from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import re

def read_corpus(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        corpus = file.readlines()
    return corpus

def preprocess_corpus(corpus):
    preprocessed_corpus = []
    for text in corpus:
        text = re.sub(r'[^\w\s]', '', text)
        preprocessed_corpus.append(text.lower())
    return preprocessed_corpus

def tokenize_corpus(corpus):
    tokenized_corpus = []
    for text in corpus:
        tokenized_corpus.append(text.split())
    return tokenized_corpus

def generate_ngrams(tokens, n):
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngrams.append(tuple(tokens[i:i+n]))
    return ngrams

def main(file_path):
    # Load the corpus
    corpus = read_corpus(file_path)

    # Preprocess the corpus
    preprocessed_corpus = preprocess_corpus(corpus)

    # Tokenize the corpus
    tokenized_corpus = tokenize_corpus(preprocessed_corpus)

    # Create n-grams for n=1, 2, 3, 4
    ngram_1 = generate_ngrams(tokenized_corpus, 1)
    ngram_2 = generate_ngrams(tokenized_corpus, 2)
    ngram_3 = generate_ngrams(tokenized_corpus, 3)
    ngram_4 = generate_ngrams(tokenized_corpus, 4)

    # Combine n-grams
    all_ngrams = ngram_1 + ngram_2 + ngram_3 + ngram_4

    # Create feature vectors using CountVectorizer
    vectorizer = CountVectorizer()
    features = vectorizer.fit_transform(all_ngrams)

    # Specify the labels for the sentiment analysis task
    sentiment_labels = [0] * len(ngram_1) + [1] * len(ngram_2) + [0] * len(ngram_3) + [1] * len(ngram_4)

    # Split the features and labels into training and test sets
    train_features, test_features, train_labels, test_labels = train_test_split(features, sentiment_labels, test_size=0.2, random_state=42)

    # Train a logistic regression model for sentiment analysis
    classifier = LogisticRegression()
    classifier.fit(train_features, train_labels)

    # Evaluate the model on the test set
    test_predictions = classifier.predict(test_features)
    accuracy = accuracy_score(test_labels, test_predictions)
    print("Accuracy:", accuracy)

# Specify the file path for the sentiment analysis task
file_path = 'GPAC.txt'
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import re

def read_corpus(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        corpus = file.readlines()
    return corpus

def preprocess_corpus(corpus):
    preprocessed_corpus = []
    for text in corpus:
        text = re.sub(r'[^\w\s]', '', text)
        preprocessed_corpus.append(text.lower())
    return preprocessed_corpus

def tokenize_corpus(corpus):
    tokenized_corpus = []
    for text in corpus:
        tokenized_corpus.append(text.split())
    return tokenized_corpus

def generate_ngrams(tokens, n):
    ngrams = []
    for token_list in tokens:
        for i in range(len(token_list) - n + 1):
            ngrams.append(" ".join(token_list[i:i+n]))
    return ngrams

def main(file_path):
    # Load the corpus
    corpus = read_corpus(file_path)

    # Preprocess the corpus
    preprocessed_corpus = preprocess_corpus(corpus)

    # Tokenize the corpus
    tokenized_corpus = tokenize_corpus(preprocessed_corpus)

    # Create n-grams for n=1, 2, 3, 4
    ngram_1 = generate_ngrams(tokenized_corpus, 1)
    ngram_2 = generate_ngrams(tokenized_corpus, 2)
    ngram_3 = generate_ngrams(tokenized_corpus, 3)
    ngram_4 = generate_ngrams(tokenized_corpus, 4)

    #  n-grams in one 
    all_ngrams = ngram_1 + ngram_2 + ngram_3 + ngram_4

 
    vectorizer = CountVectorizer()
    features = vectorizer.fit_transform(all_ngrams)

    # Specify the labels for the sentiment analysis task
    sentiment_labels = [0] * len(ngram_1) + [1] * len(ngram_2) + [0] * len(ngram_3) + [1] * len(ngram_4)

    # Split the features and labels into training and test sets
    train_features, test_features, train_labels, test_labels = train_test_split(features, sentiment_labels, test_size=0.2, random_state=42)

    # Train a logistic regression model for sentiment analysis
    classifier = LogisticRegression()
    classifier.fit(train_features, train_labels)

    # Evaluate the model on the test set
    test_predictions = classifier.predict(test_features)
    accuracy = accuracy_score(test_labels, test_predictions)
    print("Accuracy:", accuracy)


file_path = 'GPAC.txt'


main(file_path)
