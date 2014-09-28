import nltk
import string
import numpy as np


from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer

token_dict = {}
stemmer = PorterStemmer()

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

text_corpus = ["The camera has great zoom",
               "improved could be zoom improved could be zoom",
                "zoom could be improved"]

print text_corpus

corpus = []
for text in text_corpus:
    lowers_text = text.lower()
    no_punctuation_tex = lowers_text.translate(None, string.punctuation)
    corpus.append(no_punctuation_tex)

#making TF-IDF model -> this can take some time
tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
response = tfidf.fit_transform(corpus)

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


print cos_similarity(response[0], response[1])
print cos_similarity(response[1], response[2])
print cos_similarity(response[0], response[2])



#
#
def print_feature_names(response):
    for col in response.nonzero()[1]:
    #print feature_names[col], ' - ', response[0, col]