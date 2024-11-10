from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec, KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import Counter
import itertools

# Load Word2Vec model (replace 'word2vec.bin' with your model path)
# Uncomment the below line if you use a KeyedVectors model, otherwise use Word2Vec.load for models trained in Gensim
# word_emb_model = KeyedVectors.load_word2vec_format('word2vec.bin', binary=True)
word_emb_model = Word2Vec.load('word2vec.bin')  # For Gensim trained model

# TF-IDF Feature Extraction
def tfidf_features(corpus):
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_vectorizer.fit(corpus)
    feature_vectors = tfidf_vectorizer.transform(corpus)
    return feature_vectors

# Word2Vec Feature Extraction
def word2vec_features(sentence, model):
    words = [word for word in sentence.split() if word in model.wv]
    if not words:  # If none of the words are in the vocabulary, return a zero vector
        return np.zeros(model.vector_size)
    word_embeddings = [model.wv[word] for word in words]
    return np.mean(word_embeddings, axis=0)  # Mean of word embeddings

# SIF Feature Extraction
def map_word_frequency(document):
    return Counter(itertools.chain(*document))

def sif_features(sentence1, sentence2, model, a=0.001):
    sentence1 = [word for word in sentence1.split() if word in model.wv]
    sentence2 = [word for word in sentence2.split() if word in model.wv]
    word_counts = map_word_frequency([sentence1, sentence2])
    embedding_size = model.vector_size
    
    def sif_embedding(sentence):
        vs = np.zeros(embedding_size)
        for word in sentence:
            a_value = a / (a + word_counts[word])  # smooth inverse frequency
            vs = np.add(vs, np.multiply(a_value, model.wv[word]))
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
    tfidf_vectors = tfidf_features(corpus)

    # Calculate Word2Vec features
    vec1_w2v = word2vec_features(text1, model)
    vec2_w2v = word2vec_features(text2, model)

    # Calculate SIF features
    vec1_sif, vec2_sif = sif_features(text1, text2, model)

    # Cosine similarity for each method
    tfidf_similarity = get_cosine_similarity(tfidf_vectors[0].toarray(), tfidf_vectors[1].toarray())
    w2v_similarity = get_cosine_similarity(vec1_w2v, vec2_w2v)
    sif_similarity = get_cosine_similarity(vec1_sif, vec2_sif)

    print("TF-IDF Cosine Similarity:", tfidf_similarity)
    print("Word2Vec Cosine Similarity:", w2v_similarity)
    print("SIF Cosine Similarity:", sif_similarity)

# Example texts
text1 = "A girl is styling her hair."
text2 = "A girl is brushing her hair."

# Run the comparison
compare_texts(text1, text2, word_emb_model)
