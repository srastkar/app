import pymysql as mdb

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score

from operator import itemgetter

import nltk
from nltk.stem.porter import PorterStemmer

def stem_tokens(tokens):
    stemmer = PorterStemmer()
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize_and_stem(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens)
    return stems

def simple_plot(plot_X, plot_Y):
    N = len(plot_X)

    colors = np.random.rand(N)
    area = np.pi * (15 * np.random.rand(N))**2 # 0 to 15 point radiuses

    plt.scatter(plot_X, plot_Y, s=area, c=colors, alpha=0.5)
    plt.show()

def extract_features(review):
    features = []

    #features.append(np.random.rand())

    # Feature #1: length of the review
    review_length = len(review['review_text'].split())
    features.append(review_length)

    # Feature #2: star rating of the review
    features.append(review['score'])

    # Feature #3: overlap between product title and review text
    title_tokens = tokenize_and_stem(review['title'])
    review_tokens = tokenize_and_stem(review['review_text'])
    title_overlap = 0
    for token in review_tokens:
        if token in title_tokens:
            title_overlap += 1

    features.append(title_overlap)

    # Feature #4: similarity between review text and product description


    # Feature #5: number of reviews of the same reviewer in the same category
    # query = "SELECT COUNT(*) FROM reviews WHERE user_id = '" + review['user_id'] + "'"
    # cur.execute(query)
    # number_of_reviews = cur.fetchall()[0]
    # features.append(number_of_reviews)


    # Feature #6: similarity between review text and the centroid of first 3 helpful reviews

    return features

def train_model():
    plot_Y = []
    plot_X = []
    no_helpful_reviews = 0
    training_X = []
    training_y = []
    training_products = ['B00007E7JU', 'B000RZQZM0', 'B00009R6TA', 'B000NK8EWI', 'B000NP3DJW', 'B00005LEN4', 'B00009XVCZ', 'B000I1ZWRW', 'B000I7TV3W', 'B0002Y5WZM', 'B000NK3H4S',
            'B000NK6J6Q', 'B000HAOVGM', 'B0007QKN22', 'B0007QKN6S', 'B000KJQ1DG', 'B000EVLS4C', 'B00004THCZ', 'B000EMWBT2', 'B000Q30420']

    all_training_products = "','".join(training_products)

    training_data_query = "SELECT title, user_id, review_text, no_votes, no_helpful_votes, score FROM reviews WHERE product_id IN ('" + all_training_products + "') AND no_votes > 10"
    cur.execute(training_data_query)
    training_data = cur.fetchall()

    print("got all the data")

    for review in training_data:
        helpfulness_score = review['no_helpful_votes']/float(review['no_votes'])
        features = extract_features(review)
        if helpfulness_score < 0.6:
            training_y.append(0)
        else:
            training_y.append(1)
            no_helpful_reviews += 1
        print features, helpfulness_score

        training_X.append(features)

        # for plotting purposes
        # plot_Y.append(helpfulness_score)
        # plot_X.append(features[1])

    print len(training_data), no_helpful_reviews

    training_y = np.asarray(training_y)

    # Cross-validation
    scores = cross_val_score(LogisticRegression(), training_X, training_y, scoring='roc_auc', cv=10)
    print 'cross_val_score =', np.mean(scores)

    # Training
    model = LogisticRegression()
    model = model.fit(training_X, training_y)

    # Plotting
    #simple_plot(plot_X, plot_Y)

    return model


def test(model):
    test_products = ['B00004WCIC', 'B0009GZSSO', 'B000MW3YEU', 'B000Q3043Y', 'B000N386VE', 'B00091S0WA', 'B000093UDQ', 'B0002EMY9Y', 'B000KKPN5C', 'B000EXT5AY']
    for a_product in test_products:
        scored_reviews = []
        query = "SELECT review_text, no_votes, no_helpful_votes, score FROM reviews WHERE product_id IN ('" + a_product + "') AND no_votes < 2"
        cur.execute(query)
        test_data = cur.fetchall()
        for review in test_data:
            features = extract_features(review)
            predicted_helpfulness_score = model.predict_proba(features)[0][1]
            scored_reviews.append((predicted_helpfulness_score, review))

        sorted_scored_reviews = sorted(scored_reviews, key=itemgetter(0), reverse=True)

        for top_review in sorted_scored_reviews[0:20]:
            query = 'INSERT INTO scored_reviews(product_id, review_text, score) VALUES("' + a_product + '", "' + top_review[1]['review_text'] + '", "' + str(top_review[0]) + '")'
            cur.execute(query)



if __name__ == '__main__':
    con = mdb.connect('localhost', 'root', 'moosh', 'amazon') #host, user, password, #database
    cur = con.cursor(mdb.cursors.DictCursor)

    prediction_model = train_model()
    #test(prediction_model)

    con.commit()
    con.close()


