from collections.abc import Callable
import numpy as np
import math
import re


def process_query_tf_idf(query_yes, query_no):
    """Builds a tf-idf vector for two queries.
    
    Arguments
    =========
    
    query_yes: str.
    
    query_no: str.
    
    Returns
    =======
    
    processed_queries: tuple of dicts
        ({query_yes}, {query_no})
    """
    yes = re.findall(r'([a-z]+)', query_yes.lower())
    no = re.findall(r'([a-z]+)', query_no.lower())

    yes_dict = {}
    for word in yes:
        if word in yes_dict:
            yes_dict[word] += 1
        else:
            yes_dict[word] = 1

    no_dict = {}
    for word in no:
        if word in no_dict:
            no_dict[word] += 1
        else:
            no_dict[word] = 1

    return (yes_dict, no_dict)


def build_inverted_index(cars):
    """Builds an inverted index from the messages.

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

    """
    inverted = {}

    for car in cars:
        original = [car["make"].lower(), car["model"].lower()]
        for term in set(original):
            if term in inverted:
                inverted[term].append((car['id'], original.count(term)))
            else:
                inverted[term] = [(car['id'], original.count(term))]
    
    return inverted


def accumulate_dot_scores(query_counts, index, idf=None):
    """Perform a term-at-a-time iteration to efficiently compute the numerator term of cosine similarity across multiple cars.

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
    """
    car_scores = {}

    if idf is None:
        for term in query_counts:
            counts = query_counts[term]
            if counts > 0:
                q = counts
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
    """Precompute the euclidean norm of each car.
    index: the inverted index as above

    idf: dict,
        Precomputed idf values for the terms.

    n_cars: int,
        The total number of cars.
    norms: np.array, size: n_cars
        norms[j] = the norm of car j.
    """
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
            results.append((scores[car] / (q_norms * car_norms[car]), car))

        # sort the tuples by score
        results.sort(key = lambda x: x[0], reverse=True)

    else:
        for term in query:
            counts = query[term]
            if term in idf:
                q_norms += (counts * idf[term]) ** 2

        q_norms = np.sqrt(q_norms)


        # call score function
        scores = score_func(query, index, idf)

        for car in scores:
            results.append((scores[car] / (q_norms * car_norms[car]), car))


        # sort the tuples by score
        results.sort(key = lambda x: x[0], reverse=True)

    return results
