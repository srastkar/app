from flask import Flask
from flask import request
from flask import render_template
import pymysql as mdb

app = Flask(__name__)

#
# def index():
# 	return render_template("index.html",
#         title = 'Home', user = { 'nickname': 'Miguel' },
#         )

@app.route('/')
def cities_page():
    return render_template('reviews.html', reviews=[])

@app.route('/', methods=['POST', 'GET'])
def retrieve_reviews():
    product_id = request.form.get('product')
    db = mdb.connect(user="root", host="localhost", passwd="moosh", db="amazon", charset='utf8')
    cur = db.cursor()
    cur.execute("SELECT review_text, score FROM scored_reviews WHERE product_id = '"+ product_id + "' LIMIT 5;")
    query_results = cur.fetchall()
    reviews = []
    for result in query_results:
        reviews.append(dict(review_text=result[0], score=result[1]))
    return render_template('reviews.html', reviews=reviews)


if __name__ == '__main__':
    app.run()
