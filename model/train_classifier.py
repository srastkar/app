import pymysql as mdb

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from sklearn import preprocessing

from operator import itemgetter

import nltk
from nltk.stem.porter import PorterStemmer
import time

from text_compare import TF_IDF_cosine_similarity

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
    # query = "SELECT * FROM descriptions WHERE product_id = '" + review['product_id'] + "'"
    # cur.execute(query)
    # result = cur.fetchall()
    # product_desc = result[0]['description']
    # features.append(TF_IDF_cosine_similarity(review['review_text'], product_desc, []))


    # Feature #5 & #6: the reviewer activity and helpfulness
    query = "SELECT * FROM reviewers WHERE user_id = '" + review['user_id'] + "'"
    cur.execute(query)
    result = cur.fetchall()
    if len(result) == 0:
        reviewer_activity_score = 0
        reviewer_helpfulness_score = 0
    else:
        reviewer = result[0]
        reviewer_activity_score = reviewer['no_reviews']-1
        if reviewer['total_votes']-review['no_votes'] == 0:
            reviewer_helpfulness_score = 0
        else:
            reviewer_helpfulness_score = (reviewer['total_helpful_votes']-review['no_helpful_votes'])/(reviewer['total_votes']-review['no_helpful_votes'])

    features.append(reviewer_activity_score)
    features.append(reviewer_helpfulness_score)

    # Feature #7: similarity between review text and the centroid of the first 3 helpful reviews

    # converting to float so that features can be scaled
    return [float(i) for i in features]

def train_model():
    plot_Y = []
    plot_X = []
    no_helpful_reviews = 0
    training_X = []
    training_y = []

    query = "SELECT product_id from popular_recent_camera_products"
    cur.execute(query)
    products = cur.fetchall()
    test_products = []
    for product in products:
        test_products.append(product['product_id'])

    all_test_products = "','".join(test_products)

    # time > 1199145600: after 2008
    training_data_query = "SELECT * FROM reviews " \
                          "WHERE product_id NOT IN ('" + all_test_products + "') AND time > 1199145600 AND no_votes >= 10"
    cur.execute(training_data_query)
    training_data = cur.fetchall()

    print("Retrieved all the data:")
    print(str(len(training_data)) + " reviews")

    index = 0
    for review in training_data:
        print(index)
        index += 1
        helpfulness_score = review['no_helpful_votes']/float(review['no_votes'])
        features = extract_features(review)
        if helpfulness_score < 0.4:
            training_y.append(0)
            training_X.append(features)
        elif helpfulness_score > 0.7:
            training_y.append(1)
            no_helpful_reviews += 1
            training_X.append(features)
        #print features, helpfulness_scoe

    print len(training_y), no_helpful_reviews

    training_y = np.asarray(training_y)
    training_X = np.array(training_X)

    # Scaling
    min_max_scaler = preprocessing.MinMaxScaler()
    training_X_scaled = min_max_scaler.fit_transform(training_X)

    # Cross-validation
    cross_validate(training_X_scaled, training_y, ['precision', 'recall', 'f1', 'roc_auc'], 10)

    # Training
    model = LogisticRegression().fit(training_X_scaled, training_y)
    # model = RandomForestClassifier(n_estimators=1000).fit(training_X_scaled, training_y)
    # print model.feature_importances_

    return model, min_max_scaler

def cross_validate(x, y, metrics, no_of_folds):
    for metric in metrics:
        #scores = cross_val_score(RandomForestClassifier(n_estimators=50), x, y, scoring=metric, cv=no_of_folds)
        scores = cross_val_score(LogisticRegression(), x, y, scoring=metric, cv=no_of_folds)
        print metric, '=', np.mean(scores)

def test(model, scaler):
    query = "SELECT product_id FROM popular_recent_camera_products"
    cur.execute(query)
    test_products = cur.fetchall()

    for a_product in test_products:
        product_id = a_product["product_id"]
        scored_reviews = []

        # randomly retreives 200 reviews
        query = "SELECT * FROM reviews WHERE product_id IN ('" + product_id + "') ORDER BY RAND() LIMIT 200"
        cur.execute(query)
        test_data = cur.fetchall()
        for review in test_data:
            review.pop("price", None)
            features = extract_features(review)
            scaled_features = scaler.transform(np.array([features]))
            #print scaled_features
            predicted_helpfulness_score = model.predict_proba(scaled_features)[0][1]
            print predicted_helpfulness_score
            scored_reviews.append((review, predicted_helpfulness_score))

        sorted_scored_reviews = sorted(scored_reviews, key=itemgetter(1), reverse=True)

        # Saves results in a database
        for top_review in sorted_scored_reviews:
            column_values = []
            for value in top_review[0].values():
                if type(value) == str:
                    column_values.append("\""+value+"\"")
                else:
                    column_values.append(str(value))

            query = 'INSERT INTO web_scored_reviews' + \
                    "(" + ", ".join(top_review[0].keys()) + ", predicted_score" + ") " + \
                    'VALUES' + \
                    "(" + ", ".join(column_values) + ", " + str(top_review[1]) + ")"
            print query
            cur.execute(query)



if __name__ == '__main__':
    con = mdb.connect('localhost', 'root', 'moosh', 'amazon') #host, user, password, #database
    cur = con.cursor(mdb.cursors.DictCursor)

    print time.ctime()
    print('Training...')
    prediction_model, scaler = train_model()

    print time.ctime()
    print('Testing...')
    #test(prediction_model, scaler)

    print time.ctime()
    con.commit()
    con.close()


