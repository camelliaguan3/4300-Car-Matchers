import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from scipy.sparse.linalg import svds


def parse_svd_data(data, reviews):
    ''' Parse the cars data.

    Arguments
    =========

    data: the list of cars and their data

    reviews: the list of reviews 

    Returns
    =======

    cars: list of car tuples
    '''
    # we can make each car tuple consist of the car id, make, model, text, and their reviews
    cars = [(car['id'], car['make'] + ' ' + car['model'], car['text'] + ' '. join(reviews[car['id']]['reviews'])) 
            for car in data]
    
    np.random.shuffle(cars)

    return cars


def decompose(cars, k):
    ''' Create term-document matrix and decompose.

    Arguments
    =========

    cars: the list of car tuples with id, make, model, text, and reviews

    Returns
    =======

    vectorizer, word_to_index, index_to_word, u, s, v
    '''
    # maybe we can change around the max_df and min_df to get better results
    vectorizer = TfidfVectorizer(stop_words = 'english', max_df = .7, min_df = 75)

    # create the term-document matrix
    td_matrix = vectorizer.fit_transform([d[2] for d in cars])
    
    word_to_index = vectorizer.vocabulary_
    index_to_word = {i:t for t,i in word_to_index.items()}

    u, s, vt = svds(td_matrix, k=k)
    v = vt.transpose()

    return vectorizer, word_to_index, index_to_word, u, s, v


def closest_cars_to_query(query, cars, cars_compressed_normed, k):
    ''' Finds cars that are closest to the query.    

    Arguments
    =========
    query:

    cars:

    cars_compressed_normed:

    k:

    Returns
    =======

    car_list: list of cars that are closest to the query in (id, sim) format
    '''
    sims = cars_compressed_normed.dot(query)
    asort = np.argsort(-sims)[:k+1]

    car_list = [(cars[i][0], sims[i]) for i in asort[1:]]
    return car_list


def svd_closest_cars_to_query(query, cars, vectorizer, v, u, k):
    ''' Gives similarity values to cars based on svd computations.
    
    Arguments
    =========

    query: str

    cars: list of car tuples

    Returns
    =======
    
    car_list: list of cars in order of similarity
    '''
    query_tfidf = vectorizer.transform([query]).toarray()
    query_vec = normalize(np.dot(query_tfidf, v)).squeeze()

    words_compressed_normed = normalize(v, axis = 1)
    cars_compressed_normed = normalize(u, axis = 1)

    car_list = closest_cars_to_query(query_vec, cars, cars_compressed_normed, k)

    return car_list


def find_matching_terms(query, cars):
    ''' Finds the closest words to the query?
    
    Arguments
    =========
    query: str

    cars: list of car tuples

    Returns
    =======

    word_list: list of words in order of similarity
    '''
    pass