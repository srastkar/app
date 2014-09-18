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
products = {'B00004WCIC': "Canon RC-1 Wireless Remote Control for Select DSLR Cameras",
                'B0009GZSSO': "Canon Powershot S2 IS 5MP Digital Camera with 12x Optical Image Stabilized Zoom",
                'B000MW3YEU': "Panasonic Lumix DMC-TZ3S 7.2MP Digital Camera with 10x Optical Image Stabilized Zoom (Silver)",
                'B000Q3043Y': "Canon PowerShot Pro Series S5 IS 8.0MP Digital Camera with 12x Optical Image Stabilized Zoom",
                'B000N386VE': "Panasonic Lumix DMC-TZ3K 7.2MP Digital Camera with 10x Optical Image Stabilized Zoom (Black)",
                'B00091S0WA': "Sony LCS-CST General Purpose Soft Carrying Case for Slim Cybershot Digital Cameras",
                'B000093UDQ': "Digital Concepts TR-60N Camera Tripod with Carrying Case",
                'B0002EMY9Y': "Nikon SB-600 Speedlight Flash for Nikon Digital SLR Cameras",
                'B000KKPN5C': "Nikon SB-400 AF Speedlight Flash for Nikon Digital SLR Cameras",
                'B000EXT5AY': "Tamron AF 70-300mm f/4.0-5.6 Di LD Macro Zoom Lens for Konica Minolta and Sony Digital SLR Cameras"}

@app.route('/')
def landing_page():
    global products
    return render_template('index.html', products=products.values())

@app.route('/', methods=['POST', 'GET'])
def retrieve_reviews():
    #flask.request.args.get(<form_input_name>, None)
    product_title = request.form.get('product')
    product_id = [key for key in products if products[key]==product_title][0]

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
