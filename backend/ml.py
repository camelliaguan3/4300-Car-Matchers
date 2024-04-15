import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from scipy.sparse.linalg import svds


def parse_svd_data(data):
    ''' Parse the cars data.
    
    Arguments
    =========

    data: the dictionary of cars

    Returns
    =======
    
    cars: list of car tuples
    '''
    cars = [(car['id'], car['make'], car['model'], car['text'])
                for car in data]

    # np.random.shuffle(cars)

    return cars


def decompose(cars):
    ''' Create term-document matrix and decompose.
    
    Arguments
    =========

    cars: list of car tuples

    Returns
    =======
    
    several things
    '''
    vectorizer = TfidfVectorizer(stop_words = 'english', max_df = .7, min_df = 75)
    td_matrix = vectorizer.fit_transform([d[3] for d in cars])
    u, s, v_trans = svds(td_matrix, k=100)

    td_matrix_np = td_matrix.transpose().toarray()
    td_matrix_np = normalize(td_matrix_np)

    docs_comp, s, words_comp = svds(td_matrix, k=60)
    words_comp = words_comp.transpose()

    word_to_index = vectorizer.vocabulary_
    index_to_word = {i:t for t,i in word_to_index.items()}

    return (vectorizer, word_to_index, index_to_word, docs_comp, words_comp)


def svd_closest_cars_to_query(query_vec_in, docs_comp_normed, cars, k):
    ''' Gives similarity values to cars based on svd computations.
    
    Arguments
    =========

    query_vec_in: normalized tf-idf vector of query

    cars: list of car tuples

    Returns
    =======
    
    car_list: list of cars in order of similarity
    '''
    sims = docs_comp_normed.dot(query_vec_in)
    asort = np.argsort(-sims)[:k+1]

    car_list = [(cars[i][0], cars[i][1], sims[i]) for i in asort[1:]]

    return car_list