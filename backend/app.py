import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
import ml, cossim1, cossim2
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize


# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join('..',os.curdir))

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


# cosine similarity attempt with final data
def cos_search1(q_yes, q_no, min_p, max_p, no_lim, num_results):
    # handle case where the final list of cars is empty (False if empty)
    valid = True

    if q_yes != '':
        # get inverted index and process the queries to tf-idf vectors
        no_cars = len(cars_data)
        inverted_index = cossim1.build_inverted_index_final(cars_data)
        query_yes, query_no = cossim1.process_query_tf(q_yes, q_no)

        idf = None
        score_func = cossim1.accumulate_dot_scores
        car_norms = cossim1.compute_car_norms(inverted_index, no_cars, idf=None)
        results = cossim1.compute_cos_sim(query_yes, score_func, car_norms, inverted_index, idf)
        
        final = []

        for score, id in results[:num_results]:
            final.append(cars_data[id])    

        results_df = pd.DataFrame(final)

        if final == []:
            valid = False
        
    else:
        results_df = cars_df

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

        matches_filtered = matches[['make', 'model', 'year', 'starting price', 'converted car type', 'car type (epa classification)', 'color options - str', 'image', 'url']]
        # .sort_values(by='starting price', key=lambda col: col, ascending=False)
        matches_filtered_json = matches_filtered.to_json(orient='records')

    else:
        matches_filtered = pd.DataFrame([])
        matches_filtered_json = matches_filtered.to_json(orient='records')

    return matches_filtered_json


# cosine similarity attempt second prototype (Bad, needs fixing, but idk where the problem is)
def cos_search2(q_yes, q_no, min_p, max_p, no_lim, num_results):
    # handle case where the final list of cars is empty (False if empty)
    valid = True

    if q_yes != '':
        # get inverted index and process the queries to tf-idf vectors
        no_cars = len(cars_data)
        inverted_index = cossim2.build_inverted_index(cars_data)
        query_yes, query_no = cossim2.process_query_tf(q_yes, q_no)

        idf = cossim2.compute_idf(inverted_index, no_cars)
        score_func = cossim2.accumulate_dot_scores
        car_norms = cossim2.compute_car_norms(inverted_index, idf, no_cars)
        results = cossim2.compute_cos_sim(q_yes, inverted_index, idf, car_norms, score_func)
        
        final = []

        for score, id in results[:num_results]:
            print(score, id, cars_data[id], '\n')
            final.append(cars_data[id])    

        results_df = pd.DataFrame(final)

        if final == []:
            valid = False
        
    else:
        results_df = cars_df

    if valid:
        price = None
        if not no_lim:
            price = ((results_df['starting price'] > min_p) & (results_df['starting price'] < max_p))

        matches = None
        if price is None:
            matches = results_df
        else:
            matches = results_df[price]

        matches_filtered = matches[['make', 'model', 'year', 'starting price', 'converted car type', 'car type (epa classification)', 'color options - str', 'image', 'url']]
        # .sort_values(by='starting price', key=lambda col: col, ascending=False)
        matches_filtered_json = matches_filtered.to_json(orient='records')

    else:
        matches_filtered = pd.DataFrame([])
        matches_filtered_json = matches_filtered.to_json(orient='records')

    return matches_filtered_json


def svd_stuff(q_yes, q_no, min_p, max_p, no_lim, num_results):
        # handle case where the final list of cars is empty (False if empty)
    valid = True

    if q_yes != '':
        cars = ml.parse_svd_data(cars_data)
        vectorizer, word_to_ind, ind_to_word, docs_comp, words_comp = ml.decompose(cars)

        words_comp_normed = normalize(words_comp, axis = 1)
        docs_comp_normed = normalize(docs_comp)

        query_tfidf = vectorizer.transform([q_yes]).toarray()
        query_vec = normalize(np.dot(query_tfidf, words_comp)).squeeze()

        similar_cars = ml.svd_closest_cars_to_query(query_vec, docs_comp_normed, cars, num_results)

        final = []

        for id, _, _ in similar_cars:
            final.append(cars_data[id]) 

        results_df = pd.DataFrame(final)

        if final == []:
            valid = False
        
    else:
        results_df = cars_df

    if valid:
        price = None
        if not no_lim:
            price = ((results_df['starting price'] > min_p) & (results_df['starting price'] < max_p))

        matches = None
        if price is None:
            matches = results_df
        else:
            matches = results_df[price]

        matches_filtered = matches[['make', 'model', 'year', 'starting price', 'converted car type', 'car type (epa classification)', 'color options - str', 'image', 'url']]
        # .sort_values(by='starting price', key=lambda col: col, ascending=False)
        matches_filtered_json = matches_filtered.to_json(orient='records')

    else:
        matches_filtered = pd.DataFrame([])
        matches_filtered_json = matches_filtered.to_json(orient='records')

    return matches_filtered_json


@app.route('/')
def home():
    return render_template('base.html',title='')


@app.route('/cars')
def cars_search():
    text_yes  = request.args.get('yes')
    text_no   = request.args.get('no')
    min_price = request.args.get('min')
    max_price = request.args.get('max')
    no_limit  = request.args.get('no-limit')
    num_results  = int(request.args.get('num-results'))

    if not no_limit:
        max_price = int(max_price)
    min_price = int(min_price)

    return cos_search1(text_yes, text_no, min_price, max_price, no_limit, num_results)


if 'DB_NAME' not in os.environ:
    app.run(debug=True,host='0.0.0.0',port=5000)