import json
import string
from collections import deque, defaultdict

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from unidecode import unidecode
from nltk.stem import WordNetLemmatizer

# Pre-process
def pre_process(corpus):
    """Lowercases, removes stopwords, punctuation, and non-ASCII characters from text."""
    corpus = corpus.lower()
    stopset = stopwords.words('english') + list(string.punctuation)
    corpus = " ".join([i for i in word_tokenize(corpus) if i not in stopset])
    corpus = unidecode(corpus)
    return corpus

# Lemmatize function
def lemmatize_sentence(sentence):
    """Lemmatizes each word in a tokenized sentence."""
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(sentence)
    lemmatized_sentence = " ".join([lemmatizer.lemmatize(w) for w in words])
    return lemmatized_sentence

# Function to load JSON data from a file
def load_data(file_path):
    """Loads JSON data from the specified file path."""
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Function to organize comments by parent ID
def organize_comments(data):
    """Organizes comments by their parent_id."""
    comments_by_parent = defaultdict(list)
    for comment in data:
        comments_by_parent[comment["parent_id"]].append(comment)
    return comments_by_parent

# BFS traversal function with preprocessing and lemmatization
def traverse_comments_bfs(comments_by_parent):
    """Performs a BFS traversal on the comments and applies text preprocessing."""
    queue = deque(comments_by_parent[None])  # Start with root comments
    while queue:
        comment = queue.popleft()
        
        # Pre-process and lemmatize the message
        processed_message = pre_process(comment["message"])
        lemmatized_message = lemmatize_sentence(processed_message)
        
        # Output the results
        print(f'User: {comment["user"]}')
        print(f'Original Message: {comment["message"]}')
        print(f'Processed Message: {processed_message}')
        print(f'Lemmatized Message: {lemmatized_message}')
        print("----")
        
        # Add child comments to the queue
        queue.extend(comments_by_parent[comment["comment_id"]])

def main(): 
    nltk.download('punkt_tab')
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    file_path = 'src/Reddit Scraper/comments_graph.json'
    data = load_data(file_path)
    comments_by_parent = organize_comments(data)
    traverse_comments_bfs(comments_by_parent)

if __name__ == "__main__": 
    main()