import pymysql as mdb


con = mdb.connect('localhost', 'root', 'moosh', 'amazon')  # host, user, password, #database
cur = con.cursor()

f = open("/home/sarah/descriptions.txt", "r")
f.seek(1000000000)
file_text = f.read()
products = file_text.split("\n\n")
products.pop(0)
total = len(products)

count = 0
for a_product in products:
    if a_product:
        product_fields = a_product.split("\n")

    product_id = product_fields[0].split("product/productId: ")[1]
    description = product_fields[1].split("product/description: ")[1]
    description = description.replace('"', "'")

    try:
        print str(count) + " out of " + str(total)
        count += 1
        query = 'INSERT INTO descriptions(product_id, description)' + \
                'VALUES("' + product_id + '", "' + description + '")'
        cur.execute(query)
    except:
        con.commit()
        print "Unexpected error:"

con.commit()
con.close()



