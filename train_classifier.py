import pymysql as mdb

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score

from operator import itemgetter

con = mdb.connect('localhost', 'root', 'moosh', 'amazon') #host, user, password, #database
cur = con.cursor()

training_products = ['B00007E7JU', 'B000RZQZM0', 'B00009R6TA', 'B000NK8EWI', 'B000NP3DJW', 'B00005LEN4', 'B00009XVCZ', 'B000I1ZWRW', 'B000I7TV3W', 'B0002Y5WZM', 'B000NK3H4S',
            'B000NK6J6Q', 'B000HAOVGM', 'B0007QKN22', 'B0007QKN6S', 'B000KJQ1DG', 'B000EVLS4C', 'B00004THCZ', 'B000EMWBT2', 'B000Q30420']

all_training_products = "','".join(training_products)


#all_test_products = "','".join(test_products)

training_X = []
training_y = []
plot_Y = []
plot_X = []
no_helpful_reviews = 0

training_data_query = "SELECT review_text, no_votes, no_helpful_votes, score FROM reviews WHERE product_id IN ('" + all_training_products + "') AND no_votes > 10"
cur.execute(training_data_query)
training_data = cur.fetchall()


for review_text, no_votes, no_helpful_votes, score in training_data:
    helpfulness_score = no_helpful_votes/float(no_votes)
    review_length = len(review_text.split())
    plot_Y.append(helpfulness_score)
    if helpfulness_score < 0.6:
        training_y.append(0)
        no_helpful_reviews += 1
    else:
        training_y.append(1)
    training_X.append([review_length, score])
    plot_X.append(score)
    print(helpfulness_score, review_length)

print len(training_data), no_helpful_reviews	 #1116, 587

# N = 587
#
# colors = np.random.rand(N)
# area = np.pi * (15 * np.random.rand(N))**2 # 0 to 15 point radiuses
#
# plt.scatter(plot_X, plot_Y, s=area, c=colors, alpha=0.5)
# plt.show()

training_y = np.asarray(training_y)

# Cross-validation
scores = cross_val_score(LogisticRegression(), training_X, training_y, scoring='accuracy', cv=10)
print 'cross_val_score =', np.mean(scores)

# Training
model = LogisticRegression()
model = model.fit(training_X, training_y)

#Testing
test_products = ['B00004WCIC', 'B0009GZSSO', 'B000MW3YEU', 'B000Q3043Y', 'B000N386VE', 'B00091S0WA', 'B000093UDQ', 'B0002EMY9Y', 'B000KKPN5C', 'B000EXT5AY']
for a_product in test_products:
    scored_reviews = []
    query = "SELECT review_text, no_votes, no_helpful_votes, score FROM reviews WHERE product_id IN ('" + a_product + "') AND no_votes < 2"
    cur.execute(query)
    test_data = cur.fetchall()
    for review_text, no_votes, no_helpful_votes, score in test_data:
        review_length = len(review_text.split())
        helpfulness_score = model.predict_proba([review_length, score])[0][1]
        scored_reviews.append((helpfulness_score, score, no_helpful_votes, no_votes, review_text))
    #print "predicted:", helpfulness_score, "actual:", no_helpful_votes, no_votes, "review:", review_text

    sorted_scored_reviews = sorted(scored_reviews, key=itemgetter(0), reverse=True)

    for top_review in sorted_scored_reviews[0:20]:
        query = 'INSERT INTO scored_reviews(product_id, review_text, score) VALUES("' + a_product + '", "' + top_review[4] + '", "' + str(top_review[0]) + '")'
        cur.execute(query)

con.commit()
con.close()


# for review_text, no_votes, no_helpful_votes, score in test_data:
#     review_length = len(review_text.split())
#     helpfulness_score = model.predict_proba([review_length, score])[0][1]
#     scored_reviews.append((helpfulness_score, score, no_helpful_votes, no_votes, review_text))
#     #print "predicted:", helpfulness_score, "actual:", no_helpful_votes, no_votes, "review:", review_text
#
# sorted_scored_reviews = sorted(scored_reviews, key=itemgetter(0), reverse=True)
#
# for top_review in sorted_scored_reviews[1:20]:
#     print top_review
