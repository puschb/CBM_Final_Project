from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec, KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import Counter
import itertools
import requests

# Download the model files fro hugging face
def download_file(url, destination_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(destination_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    print(f"Downloaded {destination_path}")

# Word2Vec Feature Extraction
def word2vec_features(sentence, model):
    words = [word for word in sentence.split() if word in model]
    if not words:  # If none of the words are in the vocabulary, return a zero vector
        return np.zeros(model.vector_size)
    word_embeddings = [model[word] for word in words]
    return np.mean(word_embeddings, axis=0)  # Mean of word embeddings

# SIF Feature Extraction
def map_word_frequency(document):
    return Counter(itertools.chain(*document))

def sif_features(sentence1, sentence2, model, a=0.001):
    sentence1 = [word for word in sentence1.split() if word in model]
    sentence2 = [word for word in sentence2.split() if word in model]
    word_counts = map_word_frequency([sentence1, sentence2])
    embedding_size = model.vector_size
    
    def sif_embedding(sentence):
        vs = np.zeros(embedding_size)
        for word in sentence:
            a_value = a / (a + word_counts[word])  # smooth inverse frequency
            vs = np.add(vs, np.multiply(a_value, model[word]))
        if len(sentence) > 0:
            vs = np.divide(vs, len(sentence))  # weighted average
        return vs

    return sif_embedding(sentence1), sif_embedding(sentence2)

# Cosine Similarity Calculation
def get_cosine_similarity(vec1, vec2):
    return cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]

# Main comparison function
def compare_texts(text1, text2, model):
    # Preprocess and calculate TF-IDF features
    corpus = [text1, text2]

    # Calculate Word2Vec features
    vec1_w2v = word2vec_features(text1, model)
    vec2_w2v = word2vec_features(text2, model)

    # Calculate SIF features
    vec1_sif, vec2_sif = sif_features(text1, text2, model)

    # Cosine similarity for each method
    w2v_similarity = get_cosine_similarity(vec1_w2v, vec2_w2v)
    sif_similarity = get_cosine_similarity(vec1_sif, vec2_sif)

    print("Word2Vec Cosine Similarity:", w2v_similarity)
    print("SIF Cosine Similarity:", sif_similarity)

def main(download_model = False): 
    # File paths where you want to save the files
    model_path = "word2vec-google-news-300.model"
    npy_path = "word2vec-google-news-300.model.vectors.npy"

    # Download the model if we have not already
    if download_model: 
        model_url = "https://huggingface.co/fse/word2vec-google-news-300/resolve/main/word2vec-google-news-300.model"
        npy_url = "https://huggingface.co/fse/word2vec-google-news-300/resolve/main/word2vec-google-news-300.model.vectors.npy"
        download_file(model_url, model_path)
        download_file(npy_url, npy_path)

    # Load the downloaded model
    try:
        word_emb_model = KeyedVectors.load(model_path)
        print("successfully loaded model")
    except FileNotFoundError:
        print("Cannot fine model at ")

    text1 = "A girl is styling her hair."
    text2 = "A girl is brushing her hair."

    # Assuming 'model' is your loaded model
    if isinstance(word_emb_model, Word2Vec):
        print("The model is an instance of gensim.models.Word2Vec.")
        print("Use model.wv to access word vectors")
    elif isinstance(word_emb_model, KeyedVectors):
        print("The model is an instance of gensim.models.KeyedVectors.")
        print("Use model directly (not model.wv)")
    else:
        print("The model is an unknown type.")

    compare_texts(text1, text2, word_emb_model)

if __name__ == "__main__": 
    main()