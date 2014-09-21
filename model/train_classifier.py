import pymysql as mdb

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from sklearn import preprocessing

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

    # Feature #7: similarity between review text and the centroid of first 3 helpful reviews

    # converting to float so that features can be scaled
    return [float(i) for i in features]

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
        #print features, helpfulness_score

        training_X.append(features)

        # for plotting purposes
        # plot_Y.append(helpfulness_score)
        # plot_X.append(features[1])

    print len(training_data), no_helpful_reviews

    training_y = np.asarray(training_y)
    training_X = np.array(training_X)

    # Scaling
    min_max_scaler = preprocessing.MinMaxScaler()
    training_X_scaled = min_max_scaler.fit_transform(training_X)


    # Cross-validation
    scores = cross_val_score(RandomForestClassifier(), training_X_scaled, training_y, scoring='roc_auc', cv=10)
    print 'cross_val_score =', np.mean(scores)

    # Plotting ROC Curve
    X_train, X_test, y_train, y_test = train_test_split(training_X_scaled, training_y, test_size=.5, random_state=0)
    predicted_y = RandomForestClassifier().fit(X_train, y_train).predict_proba(X_test)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, predicted_y[:,1], pos_label=1)

    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = 0.91)')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    # Plotting
    #simple_plot(plot_X, plot_Y)

    # Training
    model = RandomForestClassifier(n_estimators=10000).fit(training_X_scaled, training_y)
    print model.feature_importances_

    return model, min_max_scaler


def test(model, scaler):
    test_products = ['B00004WCIC', 'B0009GZSSO', 'B000MW3YEU', 'B000Q3043Y', 'B000N386VE', 'B00091S0WA', 'B000093UDQ', 'B0002EMY9Y', 'B000KKPN5C', 'B000EXT5AY']
    for a_product in test_products:
        scored_reviews = []
        query = "SELECT * FROM reviews WHERE product_id IN ('" + a_product + "')"
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

        for top_review in sorted_scored_reviews[0:20]:
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
            #print query
            cur.execute(query)



if __name__ == '__main__':
    con = mdb.connect('localhost', 'root', 'moosh', 'amazon') #host, user, password, #database
    cur = con.cursor(mdb.cursors.DictCursor)

    print('Training...')
    prediction_model, scaler = train_model()

    print('Testing...')
    test(prediction_model, scaler)

    con.commit()
    con.close()


