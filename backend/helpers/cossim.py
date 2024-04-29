import numpy as np
import re


def process_query_tf(query_yes):
    ''' Builds a tf vector for queries.
    
    Arguments
    =========
    
    query_yes: str.
        
    Returns
    =======
    
    processed_queries: dict
    '''

    # MAKE SURE THIS DOESN'T JUST YEET ALL NON LETTERS
    yes = re.findall(r'([0-9a-z]+)', query_yes.lower())
    yes_dict = {}
    for word in yes:
        if word in yes_dict:
            yes_dict[word] += 1
        else:
            yes_dict[word] = 1

    return yes_dict


def build_inverted_index_basic(cars):
    ''' Builds an inverted index from the messages.

    Arguments
    =========

    msgs: list of dicts.

    Returns
    =======

    inverted_index: dict
        For each term, the index contains
        a sorted list of tuples (id, count_of_term_in_car)
        such that tuples with smaller ids appear first:
        inverted_index[term] = [(d1, tf1), (d2, tf2), ...]

    '''
    inverted = {}

    for car in cars:
        # TRY TO HANDLE CARS WITH LIKE DASHES OR PLUSSES OR SOMETHING
        original = car['make'].lower().replace('-', ' ').split() + car['model'].lower().replace('-', ' ').split()
        for term in set(original):
            if term in inverted:
                inverted[term].append((car['id'], original.count(term)))
            else:
                inverted[term] = [(car['id'], original.count(term))]

    return inverted


# UPDATE TO INCLUDE OTHER SPECS
def build_inverted_index_final(cars):
    ''' Builds an inverted index from the messages.

    Arguments
    =========

    msgs: list of dicts.

    Returns
    =======

    inverted_index: dict
        For each term, the index contains
        a sorted list of tuples (id, count_of_term_in_car)
        such that tuples with smaller ids appear first:
        inverted_index[term] = [(d1, tf1), (d2, tf2), ...]

    '''
    inverted = {}

    for car in cars:
        # TRY TO HANDLE CARS WITH LIKE DASHES OR PLUSSES OR SOMETHING
        colors = []
        for color in car['standardized colors']:
            if color != 'Tbd':
                colors.append(color.lower())
        
        # for color in car['color options']:
        #     colors += color.lower().split()

        original = (car['make'].lower().replace('-', ' ').split() 
                    + car['model'].lower().replace('-', ' ').split()
                    + colors
                    + car['converted car type'].lower().split()
                    + car['car type (epa classification)'].lower().split())
        
        for term in set(original):
            if term in inverted:
                inverted[term].append((car['id'], original.count(term)))
            else:
                inverted[term] = [(car['id'], original.count(term))]

    return inverted


def accumulate_dot_scores(query_counts, index, idf=None):
    ''' Perform a term-at-a-time iteration to efficiently compute the numerator 
        term of cosine similarity across multiple cars.

    Arguments
    =========

    query_counts: dict,
        tf-idf

    index: the inverted index as above,

    idf: dict,
        Precomputed idf values for the terms.

    Returns
    =======

    car_scores: dict
        Dictionary mapping from car ID to the final accumulated score for that car
    '''
    car_scores = {}

    if idf is None:
        for term in query_counts:
            counts = query_counts[term]
            if counts > 0:
                q = counts
                if term in index:
                    for tup in index[term]:
                        car_no = tup[0]
                        tf = tup[1]

                        if car_no in car_scores:
                            car_scores[car_no] += q * tf
                        else:
                            car_scores[car_no] = q * tf
    else:
        for term in query_counts:
            counts = query_counts[term]
            if counts > 0 and term in idf:
                q = counts * idf[term]
                for tup in index[term]:
                    car_no = tup[0]
                    tf = tup[1]

                    if car_no in car_scores:
                        car_scores[car_no] += q * tf * idf[term]
                    else:
                        car_scores[car_no] = q * tf * idf[term]

    return car_scores


def compute_car_norms(index, n_cars, idf=None):
    ''' Precompute the euclidean norm of each car.
    
    Arguments
    =========

    index: the inverted index as above

    idf: dict,
        Precomputed idf values for the terms.

    n_cars: int,
        The total number of cars.

    Returns
    =======

    norms: np.array, size: n_cars
        norms[j] = the norm of car j.
    '''
    norms = np.zeros((n_cars))
    
    if idf is None:
        for term in index:
            lst = index[term]
            for tup in lst:
                car_no = tup[0]
                tf = tup[1]

                norms[car_no] += (tf) ** 2

    else:
        for term in index:
            lst = index[term]
            if term in idf:
                idf_term = idf[term]

                for tup in lst:
                    car_no = tup[0]
                    tf = tup[1]

                    norms[car_no] += (tf * idf_term) ** 2

    return np.sqrt(norms)


def compute_cos_sim(query, score_func, car_norms, index, idf=None):
    ''' Precompute the cosine similarity of each car to the query.
    
    Arguments
    =========

    query: the tf-idf vectorized query

    car_norms: np.array, size: n_cars
        norms[j] = the norm of car j.

    index: the inverted index as above

    idf: dict,
        Precomputed idf values for the terms.

    Returns
    =======
    
    cos_sim: list of tuples (id, score)
    '''
    results = []

    # calculate the norm for q
    q_norms = 0

    if idf is None:
        for term in query:
            counts = query[term]
            q_norms += (counts) ** 2

        q_norms = np.sqrt(q_norms)

        # call score function
        scores = score_func(query, index, idf)

        for car in scores:
            score = scores[car] / (q_norms * car_norms[car])
            results.append((car, score))


        # sort the tuples by score
        results.sort(key = lambda x: x[1], reverse=True)

    else:
        for term in query:
            counts = query[term]
            if term in idf:
                q_norms += (counts * idf[term]) ** 2

        q_norms = np.sqrt(q_norms)


        # call score function
        scores = score_func(query, index, idf)

        for car in scores:
            score = scores[car] / (q_norms * car_norms[car])
            results.append((car, score))


        # sort the tuples by score
        results.sort(key = lambda x: x[1], reverse=True)

    return results