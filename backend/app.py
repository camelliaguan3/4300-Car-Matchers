import os
import json, re
from flask import Flask, render_template, request
from flask_cors import CORS
import helpers.ml as ml, helpers.cossim as cossim
import pandas as pd
from operator import itemgetter



# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join('..',os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script
json_file_path = os.path.join(current_directory, 'init.json')



# Helper functions for processing json
def split_lower_words(sentence, splitter):
    ''' Splits and lowers words in a sentence.
    
    Arguments
    =========
    
    sentence: string

    Returns
    =======

    lower_l: list of words (all lowercase)
    '''
    lower_l = [w.lower() for w in splitter.findall(sentence)]
    return lower_l


def get_most_common_words(review):
    ''' Finds the most common words within the reviews.

    Arguments
    =========

    review: list of sentences

    Returns
    =======

    lowered_sentences: list of lists of words
    '''

    review = ' '.join(review)
    review_edited = re.sub(r"\[\d+\]", "", review)

    sentence_splitter = re.compile(r'(?<![A-Z])[.!?](?=\s+[A-Z])', re.VERBOSE)
    word_splitter = re.compile(r'(\w+)', re.VERBOSE)

    lowered_sentences = [split_lower_words(sentence, word_splitter) for sentence in sentence_splitter.split(review_edited)]

    IDF = {}
    DF = {}

    words = [w for sentence in lowered_sentences for w in sentence]
    terms = sorted(set(words))

    for t in terms:
        DF[t] = len([1 for sentence in lowered_sentences if t in sentence])
        IDF[t] = 1 / float(DF[t] + 1)

    common_words = []
    for IDF_t in sorted(IDF.items(), key=itemgetter(1),reverse = True):
        common_words.append(IDF_t)

    return common_words


def process_reviews_and_cars(cars_data, reviews_data):
    ''' Combines the cars and their reviews

    Arguments
    =========

    cars_data: list of car dictionaries

    reviews_data: list of review dictionaries

    Returns
    =======

    combined: list of cars dictionaries with reviews and ratings
    '''
    combined = []
    
    for car in cars_data:
        id = car['id']
        review = list(filter(lambda r: r['id'] == id, reviews_data))[0]

        rating = review['rating']
        reviews = review['reviews']

        car['rating'] = rating
        car['reviews'] = reviews
        car['score'] = 0

        # [(word, idf), ...]
        car['common'] = get_most_common_words(reviews)

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
def cos_search(q, min_p, max_p, no_lim, num_results):
    # handle case where the final list of cars is empty (False if empty)
    valid = True

    if q != '':
        # get inverted index and process the queries to tf-idf vectors
        no_cars = len(combined_data)
        inverted_index = cossim.build_inverted_index_final(combined_data)
        query = cossim.process_query_tf(q)

        idf = None
        score_func = cossim.accumulate_dot_scores
        car_norms = cossim.compute_car_norms(inverted_index, no_cars, idf=None)
        results = cossim.compute_cos_sim(query, score_func, car_norms, inverted_index, idf)
        
        final = []

        for id, score in results:
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



# svd combined with cosine similarity
def combine_svd_w_cos_search(q, min_p, max_p, no_lim, num_results):
    # handle case where the final list of cars is empty (False if empty)
    valid = True
    k = 60

    if q != '':
        # cossim
        no_cars = len(combined_data)
        inverted_index = cossim.build_inverted_index_final(combined_data)
        query = cossim.process_query_tf(q)

        idf = None
        score_func = cossim.accumulate_dot_scores
        car_norms = cossim.compute_car_norms(inverted_index, no_cars, idf=None)
        results_cos = cossim.compute_cos_sim(query, score_func, car_norms, inverted_index, idf)
        

        # svd
        cars = ml.parse_svd_data(cars_data, reviews_data)
        vectorizer, word_to_index, index_to_word, u, s, v = ml.decompose(cars, k)
        results_svd = ml.svd_closest_cars_to_query(q, cars, vectorizer, v, u, k)
        
        word_splitter = re.compile(r'(\w+)', re.VERBOSE)
        query_list = [w.lower() for w in word_splitter.findall(q)]

        final = []
        combined = []


        # combine results from cossim and svd
        if results_cos != [] and results_svd == []:
            combined = results_cos

        elif results_cos == [] and results_svd != []:
            combined = results_svd

        elif results_cos != [] and results_svd != []:
            for id in range(len(combined_data)):
                score = 0
                
                for id_cos, score_cos in results_cos:
                    if id == id_cos:
                        score += score_cos * 0.8
                        break

                for id_svd, score_svd in results_svd:
                    if id == id_svd:
                        score += score_svd * 0.2
                        break
            
                combined.append((id, score))

            combined.sort(key = lambda x: x[1], reverse=True)


        # put cars into final list
        for id, score in combined:
            c = combined_data[id]
            c['score'] = score
            if score > 0:
                final.append(c) 


        # find common words to query
        for car in final:
            common_words_w_query = []
            for word in query_list:
                if word in car['common']:
                    common_words_w_query.append(word)

            c['common words'] = common_words_w_query


        results_df = pd.DataFrame(final)

        if final == []:
            valid = False
        
    else:
        results_df = combined_df


    if valid:
        # filter via price
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

        matches_filtered = matches[['make', 'model', 'year', 'starting price', 'converted car type', 'car type (epa classification)', 'color options - str', 'image', 'url', 'rating', 'reviews', 'score']]
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
    text  = request.args.get('yes')
    min_price = request.args.get('min')
    max_price = request.args.get('max')
    no_limit  = request.args.get('no-limit')
    num_results  = int(request.args.get('num-results'))

    if not no_limit:
        max_price = int(max_price)
    min_price = int(min_price)


    return combine_svd_w_cos_search(text, min_price, max_price, no_limit, num_results)


if 'DB_NAME' not in os.environ:
    app.run(debug=True,host='0.0.0.0',port=5000)