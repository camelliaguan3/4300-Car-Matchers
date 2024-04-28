import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from scipy.sparse.linalg import svds



# How to use SVD: 
# svd to decompose matrix of documents into singular values, gives basis + singular values
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
    cars = [(car['id'], car['make'], car['model'], car['text'], reviews[car['id']]['reviews']) 
            for car in data]
    
    np.random.shuffle(cars)

    return cars


# select basis, change vector space to those selected basis values.

# decompose?

def decompose(cars):
    ''' Create term-document matrix and decompose.

    Arguments
    =========

    cars: the list of car tuples with id, make, model, text, and reviews

    Returns
    =======

    idk
    
    '''
    # maybe we can change around the max_df and min_df to get better results
    vectorizer = TfidfVectorizer(stop_words = 'english', max_df = .7, min_df = 75)

    # create the term-document matrix
    td_matrix = vectorizer.fit_transform([d[3] + ' '.join(d[4]) for d in cars])
    
    # (sentence_index, feature_index) count -> td_matrix format
    # print('td', td_matrix)

    word_to_index = vectorizer.vocabulary_
    docs_comp, s, words_comp = svds(td_matrix, k=100)
    words_comp = words_comp.transpose()

    word_to_index = vectorizer.vocabulary_
    index_to_word = {i: t for t, i in word_to_index.items()}

    words_comp_normed = normalize(words_comp, axis=1)
    docs_comp_normed = normalize(docs_comp, axis=1)
    return vectorizer, word_to_index, index_to_word, words_comp_normed, docs_comp_normed


# do cosine similarity in new vector space, use matrix from decomposition, map queries and documents into the new space
def svd_closest_words(query, index_to_word, word_to_index, words_comp_normed, k):
    if query not in word_to_index:
        return None
    
    sims = words_comp_normed.dot(words_comp_normed[word_to_index[query], :])
    asort = np.argsort(-sims)[k:1]
    return [(index_to_word[i], sims[i]) for i in asort[1:]]


def svd_closest_cars_to_query(query, cars, num_results):
    ''' Gives similarity values to cars based on svd computations.
    
    Arguments
    =========

    query: str

    cars: list of car tuples

    Returns
    =======
    
    car_list: list of cars in order of similarity
    '''
        
    vectorizer, word_to_index, index_to_word, words_comp_normed, docs_comp_normed = decompose(cars)

    query_list = query.split()
