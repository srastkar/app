import pymysql as mdb

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from pylab import *


def review_length_box_plot():
    # get data
    con = mdb.connect('localhost', 'root', 'moosh', 'amazon')
    cur = con.cursor(mdb.cursors.DictCursor)
    query = "SELECT product_id from popular_2012_camera_products LIMIT 20"
    cur.execute(query)
    products = cur.fetchall()
    training_products = []
    for product in products:
        training_products.append(product['product_id'])

    all_training_products = "','".join(training_products)

    training_data_query = "SELECT title, user_id, review_text, no_votes, no_helpful_votes, score FROM reviews WHERE product_id IN ('" + all_training_products + "') AND no_votes > 10"
    cur.execute(training_data_query)
    reviews = cur.fetchall()

    # create input lists for the plotting function
    bin_100 = []
    bin_200 = []
    bin_300 = []
    bin_over_300 = []
    for review in reviews:
        review_length = len(review['review_text'].split())
        review_helpfulness = review['no_helpful_votes']/float(review['no_votes'])
        if review_length < 100:
            bin_100.append(review_helpfulness)
        elif review_length < 200:
            bin_200.append(review_helpfulness)
        elif review_length < 300:
            bin_300.append(review_helpfulness)
        else:
            bin_over_300.append(review_helpfulness)

    # plot
    plt.figure()
    plt.ylim([-0.05, 1.05])
    plt.boxplot([bin_100, bin_200, bin_300, bin_over_300], 0)
    plt.show()

def plot_roc():
    # # Plotting ROC Curve
    # X_train, X_test, y_train, y_test = train_test_split(training_X_scaled, training_y, test_size=.5, random_state=0)
    # predicted_y = RandomForestClassifier().fit(X_train, y_train).predict_proba(X_test)
    # fpr, tpr, thresholds = metrics.roc_curve(y_test, predicted_y[:,1], pos_label=1)
    #
    # plt.figure()
    # plt.plot(fpr, tpr, label='ROC curve (area = )')
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic example')
    # plt.legend(loc="lower right")
    # plt.show()


def simple_plot(plot_X, plot_Y):
    mpl.rcParams['axes.labelsize'] = 'large'
    mpl.rcParams['font.size'] = '20'

    N = len(plot_X)

    colors = np.random.rand(N)
    area = np.pi * (15 * np.random.rand(N))**2 # 0 to 15 point radiuses

    plt.scatter(plot_X, plot_Y, s=area, c=colors, alpha=0.5)
    plt.show()


review_length_box_plot()


