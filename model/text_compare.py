import nltk
import string
import numpy as np


from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stemmer = PorterStemmer()
    stems = stem_tokens(tokens, stemmer)
    return stems


# vec1 & vec2 are sparse SciPy matrices
def dot_product(vec1, vec2):
    vec1_cols = vec1.nonzero()[1].tolist()
    vec2_cols = vec2.nonzero()[1].tolist()

    common_cols = set(vec1_cols).intersection(vec2_cols)

    dot_product = 0
    for col in common_cols:
         dot_product += vec1.getcol(col)[0,0]*vec2.getcol(col)[0,0]

    return dot_product


# each input is a sparse word-document TF-IDF matrix
def cos_similarity(doc1, doc2):
     return dot_product(doc1, doc2) / (np.sqrt(dot_product(doc1, doc1)) * np.sqrt(dot_product(doc2, doc2)))

#
# def print_feature_names(response):
#     for col in response.nonzero()[1]:
#         print feature_names[col], ' - ', response[0, col]

def TF_IDF_cosine_similarity(text1, text2, corpus):
    IDF_corpus = [text1, text2]
    IDF_corpus.extend(corpus)
    print IDF_corpus
    preprocessed_corpus = []
    for text in IDF_corpus:
        lowers_text = text.lower()
        no_punctuation_text = lowers_text.translate(None, string.punctuation)
        preprocessed_corpus.append(no_punctuation_text)

    #making TF-IDF model -> this can take some time
    tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
    response = tfidf.fit_transform(corpus)

    return cos_similarity(response[0], response[1])
