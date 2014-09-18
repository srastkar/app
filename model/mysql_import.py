import pymysql as mdb

		
con = mdb.connect('localhost', 'root', 'moosh', 'amazon') #host, user, password, #database
cur = con.cursor()

f = open("/home/sarah/Electronics.txt", "r")
file_text = f.read()

reviews = file_text.split("\n\n")


count = 0
for a_review in reviews:
	if a_review:
		review_fields = a_review.split("\n")
	
		product_id = review_fields[0].split("product/productId: ")[1]
        	title = review_fields[1].split("product/title: ")[1]
		title = title.replace('"', "'")
		price = review_fields[2].split("product/price: ")[1]
		user_id = review_fields[3].split("review/userId: ")[1]
		profile_name = review_fields[4].split("review/profileName: ")[1]
		profile_name = profile_name.replace('"', "'")
		helpfulness = review_fields[5].split("review/helpfulness: ")[1]
		no_helpful_votes = helpfulness.split("/")[0]
		no_votes = helpfulness.split("/")[1]
		score = review_fields[6].split("review/score: ")[1]
		time = review_fields[7].split("review/time: ")[1]
		summary = review_fields[8].split("review/summary: ")[1]
		summary = summary.replace('"', "'")
		review_text = review_fields[9].split("review/text: ")[1]
		review_text = review_text.replace('"', "'")

		#print("Review:")
		#print(product_id, title, price, user_id, profile_name, no_votes, no_helpful_votes, score, time, summary, review_text)
		#print("Insert into database:")
		
		try:
			print count, "reviews:"
			count += 1			
			query = 'INSERT INTO reviews(product_id, title, price, user_id, profile_name, no_votes, no_helpful_votes, score, time, summary, review_text) ' + \
				'VALUES("' + product_id + '", "' + title + '", "' + price + '", "' + user_id + '", "' + profile_name + '", "' + no_votes + '", "' + \
				no_helpful_votes + '", "' + score + '", "' + time + '", "' + summary + '", "' + review_text + '")'
				
			#print("Query:")			
			#print(query)
			cur.execute(query)
		except:
			con.commit()
			print "Unexpected error:"
			
con.commit()
con.close()
    		
				


#with con: 
#    cur = con.cursor()
#    cur.execute("SELECT COUNT(*) FROM reviews")
#    rows = cur.fetchall()
#    for row in rows:
#        print row
