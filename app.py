from flask import Flask
from flask import request
from flask import render_template
import pymysql as mdb

app = Flask(__name__)

@app.route('/')
def landing_page():
    query = "SELECT title from popular_recent_camera_products order by no_reviews desc limit 50"
    cur.execute(query)
    test_products = cur.fetchall()
    product_titles = []
    for product in test_products:
        product_titles.append(product['title'])
    return render_template('index.html', products=product_titles)

@app.route('/', methods=['POST', 'GET'])
def retrieve_reviews():
    #flask.request.args.get(<form_input_name>, None)
    product_title = request.form.get('product')
    query = "SELECT product_id FROM popular_recent_camera_products WHERE title = '" + product_title + "'"
    cur.execute(query)
    product_id = cur.fetchall()[0]["product_id"]
    cur.execute("SELECT * FROM web_scored_reviews WHERE product_id = '"+ product_id + "' AND no_votes < 5 ORDER BY predicted_score DESC limit 20") #   AND predicted_score < 1")
    query_results = cur.fetchall()
    return render_template('reviews.html', reviews=query_results)


if __name__ == '__main__':
    db = mdb.connect(user="root", host="localhost", passwd="moosh", db="amazon", charset='utf8')
    cur = db.cursor(mdb.cursors.DictCursor)
    app.run()
