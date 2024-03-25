import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import helper
import numpy as np
import pandas as pd

# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script
json_file_path = os.path.join(current_directory, 'init.json')

# Assuming your JSON data is stored in a file named 'init.json'
with open(json_file_path, 'r') as file:
    data = json.load(file)
    cars_data = data['cars']
    cars_df = pd.DataFrame(data['cars'])
    reviews_df = pd.DataFrame(data['reviews'])

app = Flask(__name__)
CORS(app)


def json_search_reviews(query):
    matches = []
    merged_df = pd.merge(cars_df, reviews_df, left_on='id', right_on='id', how='inner')
    matches = merged_df[merged_df['title'].str.lower().str.contains(query.lower())]
    matches_filtered = matches[['title', 'descr', 'imdb_rating']]
    matches_filtered_json = matches_filtered.to_json(orient='records')
    return matches_filtered_json


# Sample search using json with pandas
def json_search(q_yes, q_no, min_p, max_p, no_lim):
    matches = []
    
    # searching without reviews first
    make = cars_df['make'].str.lower().str.contains(q_yes.lower())

    price = None
    if not no_lim:
        price = ((cars_df['starting price'] > int(min_p)) & (cars_df['starting price'] < int(max_p)))

    matches = None
    if price is None:
        matches = cars_df[make]
    else:
        matches = cars_df[make & price]

    # print(matches)
    matches_filtered = matches[['make', 'model', 'year', 'starting price', 'image', 'url']].sort_values(by='starting price', key=lambda col: col)
    matches_filtered_json = matches_filtered.to_json(orient='records')
    return matches_filtered_json



# cosine similarity attempt
def cos_search(q_yes, q_no, min_p, max_p, no_lim, num_results):

    if q_yes != '':

        # get inverted index and process the queries to tf-idf vectors
        no_cars = len(cars_data)
        inverted_index = helper.build_inverted_index(cars_data)
        query_yes, query_no = helper.process_query_tf_idf(q_yes, q_no)

        idf = None
        score_func = helper.accumulate_dot_scores
        car_norms = helper.compute_car_norms(inverted_index, no_cars, idf=None)
        results = helper.compute_cos_sim(query_yes, score_func, car_norms, inverted_index, idf)
        
        final = []

        for score, id in results[:num_results]:
            final.append(cars_data[id])    

        results_df = pd.DataFrame(final)
        
    else:
        results_df = cars_df
    
    # searching without reviews first

    price = None
    if not no_lim:
        price = ((results_df['starting price'] > min_p) & (results_df['starting price'] < max_p))

    matches = None
    if price is None:
        matches = results_df
    else:
        matches = results_df[price]

    # print(matches)
    matches_filtered = matches[['make', 'model', 'year', 'starting price', 'image', 'url']].sort_values(by='starting price', key=lambda col: col, ascending=False)
    matches_filtered_json = matches_filtered.to_json(orient='records')
    return matches_filtered_json


@app.route("/")
def home():
    return render_template('base.html',title="")

@app.route("/cars")
def cars_search():
    text_yes  = request.args.get("yes")
    text_no   = request.args.get("no")
    min_price = request.args.get("min")
    max_price = request.args.get("max")
    no_limit  = request.args.get("no-limit")
    num_results  = int(request.args.get("num-results"))

    if not no_limit:
        min_price = int(min_price)
        max_price = int(max_price)

    return cos_search(text_yes, text_no, min_price, max_price, no_limit, num_results)


if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)