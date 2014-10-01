from flask import Flask
from flask import request
from flask import render_template
import pymysql as mdb
import datetime

app = Flask(__name__)

@app.route('/')
def landing_page():
    query = "SELECT title from popular_recent_camera_products order by no_reviews desc limit 200"
    cur.execute(query)
    test_products = cur.fetchall()
    product_titles = []
    for product in test_products:
        product_titles.append(product['title'])
    return render_template('index.html', products=product_titles)

@app.route('/', methods=['POST', 'GET'])
def retrieve_reviews():
    product_title = request.form.get('product')
    query = "SELECT product_id FROM popular_recent_camera_products WHERE title = '" + product_title + "'"
    cur.execute(query)
    product_id = cur.fetchall()[0]["product_id"]
    cur.execute("SELECT * FROM web_scored_reviews WHERE product_id = '"+ product_id + "' ORDER BY predicted_score DESC") #AND AND predicted_score < 1 no_votes < 5  DESC LIMIT 20    ")
    query_results = cur.fetchall()
    for review in query_results:
        review['time'] = datetime.datetime.fromtimestamp(review['time']).strftime('%B %d, %Y')
    return render_template('reviews.html', reviews=query_results, title=product_title)

@app.route('/slides')
def slides():
    return render_template('slides.html')

if __name__ == '__main__':
    db = mdb.connect(user="root", host="localhost", passwd="moosh", db="amazon", charset='utf8')
    cur = db.cursor(mdb.cursors.DictCursor)
    app.run(debug=True)
