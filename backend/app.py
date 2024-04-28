import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
import helpers.ml as ml, helpers.cossim1 as cossim1
import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import normalize


# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join('..',os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script
json_file_path = os.path.join(current_directory, 'init.json')


def process_reviews_and_cars(cars_data, reviews_data):
    combined = []
    
    for car in cars_data:
        id = car['id']
        review = list(filter(lambda r: r['id'] == id, reviews_data))[0]

        rating = review['rating']
        reviews = review['reviews']

        car['rating'] = rating
        car['reviews'] = reviews
        combined.append(car)

    return combined


# Assuming your JSON data is stored in a file named 'init.json'
with open(json_file_path, 'r') as file:
    data = json.load(file)
    cars_data = data['cars']
    reviews_data = data['reviews']
    combined_data = process_reviews_and_cars(cars_data, reviews_data)
    cars_df = pd.DataFrame(cars_data)
    reviews_df = pd.DataFrame(reviews_data)
    combined_df = pd.DataFrame(combined_data)


app = Flask(__name__)
CORS(app)


# cosine similarity attempt with final data
def cos_search1(q, min_p, max_p, no_lim, num_results):
    # handle case where the final list of cars is empty (False if empty)
    valid = True

    if q != '':
        # get inverted index and process the queries to tf-idf vectors
        no_cars = len(combined_data)
        inverted_index = cossim1.build_inverted_index_final(combined_data)
        query = cossim1.process_query_tf(q)

        idf = None
        score_func = cossim1.accumulate_dot_scores
        car_norms = cossim1.compute_car_norms(inverted_index, no_cars, idf=None)
        results = cossim1.compute_cos_sim(query, score_func, car_norms, inverted_index, idf)
        
        final = []

        for score, id in results:
            final.append(combined_data[id])    

        results_df = pd.DataFrame(final)

        if final == []:
            valid = False
        
    else:
        results_df = combined_df

    if valid:
        price = None
        if not no_lim:
            price = ((results_df['starting price'] > min_p) & (results_df['starting price'] < max_p))
        else:
            price = (results_df['starting price'] > min_p)

        matches = None
        if price is None:
            matches = results_df
        else:
            matches = results_df[price]

        matches_filtered = matches[['make', 'model', 'year', 'starting price', 'converted car type', 'car type (epa classification)', 'color options - str', 'image', 'url', 'rating', 'reviews']]
        # .sort_values(by='starting price', key=lambda col: col, ascending=False)
        matches_filtered_json = matches_filtered.to_json(orient='records')

    else:
        matches_filtered = pd.DataFrame([])
        matches_filtered_json = matches_filtered.to_json(orient='records')

    return matches_filtered_json


def svd_cossim(q, min_p, max_p, no_lim, num_results):
    query = cossim1.process_query_tf(q)

    q = re.findall(r'([0-9a-z]+)', q.lower())

    cars = ml.parse_svd_data(cars_data, reviews_data)
    sims = ml.svd_closest_cars_to_query(' '.join(q), cars, num_results)
    
    

@app.route('/')
def home():
    return render_template('base.html',title='')


@app.route('/cars')
def cars_search():
    text  = request.args.get('yes')
    min_price = request.args.get('min')
    max_price = request.args.get('max')
    no_limit  = request.args.get('no-limit')
    num_results  = int(request.args.get('num-results'))

    if not no_limit:
        max_price = int(max_price)
    min_price = int(min_price)

    svd_cossim(text, min_price, max_price, no_limit, num_results)

    return cos_search1(text, min_price, max_price, no_limit, num_results)


if 'DB_NAME' not in os.environ:
    app.run(debug=True,host='0.0.0.0',port=5000)