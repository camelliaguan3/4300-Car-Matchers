import numpy as np
import re
import math
from nltk.tokenize import TreebankWordTokenizer

treebank_tokenizer = TreebankWordTokenizer()


def process_query_tf(query_yes, query_no):
    ''' Builds a tf vector for two queries.

    Returns
    =======
    
    processed_queries: tuple of dicts
        ({query_yes}, {query_no})
    '''

    # MAKE SURE THIS DOESN'T JUST YEET ALL NON LETTERS
    yes = re.findall(r'([0-9a-z]+)', query_yes.lower())
    no = re.findall(r'([0-9a-z]+)', query_no.lower())

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
    ''' Builds an inverted index from the cars.

    Returns
    =======

    inverted_index: dict
        For each term, the index contains
        a sorted list of tuples (id, count_of_term_in_car)
        such that tuples with smaller ids appear first:
        inverted_index[term] = [(d1, tf1), (d2, tf2), ...]

    '''
    inverted = {}

    for id in range(len(cars)):
        original = cars[id]['text'].lower().replace('-', ' ').replace(', ', ' ').replace(',', ' ').split()

        for term in set(original):
            if term in inverted:
                inverted[term].append((id, original.count(term)))
            else:
                inverted[term] = [(id, original.count(term))]
    
    return inverted


def compute_idf(inv_idx, n_docs, min_df=5, max_df_ratio=0.95):
    """ Compute term IDF values from the inverted index.

    Returns
    =======

    idf: dict
        For each term, the dict contains the idf value.

    """
    idf = {}
    
    for term in inv_idx:
        df = len(inv_idx[term])
        if df >= min_df and (df / n_docs) <= max_df_ratio:
            idf[term] = math.log(n_docs / (1 + df), 2)

    return idf


def compute_car_norms(index, idf, n_docs):
    ''' Precompute the euclidean norm of each car.

    Returns
    =======

    norms: np.array, size: n_cars
        norms[j] = the norm of car j.
    '''
    norms = np.zeros((n_docs))
    
    for term in index:
        lst = index[term]
        if term in idf:
            idf_term = idf[term]

            for tup in lst:
                doc_no = tup[0]
                tf = tup[1]

                norms[doc_no] += (tf * idf_term) ** 2

    return np.sqrt(norms)


def accumulate_dot_scores(query_word_counts, index, idf):
    """ Perform a term-at-a-time iteration to efficiently compute the numerator 
        term of cosine similarity across multiple documents.

    Returns
    =======

    doc_scores: dict
        Dictionary mapping from doc ID to the final accumulated score for that doc
    """
    doc_scores = {}

    for term in query_word_counts:
        counts = query_word_counts[term]
        if counts > 0 and term in idf:
            q = counts * idf[term]
            for tup in index[term]:
                doc_no = tup[0]
                tf = tup[1]

                if doc_no in doc_scores:
                    doc_scores[doc_no] += q * tf * idf[term]
                else:
                    doc_scores[doc_no] = q * tf * idf[term]

    return doc_scores


def compute_cos_sim(query, index, idf, doc_norms, score_func, tokenizer=treebank_tokenizer):
    """ Search the collection of documents for the given query

    Returns
    =======

    results, list of tuples (score, doc_id)
        Sorted list of results such that the first element has
        the highest score, and `doc_id` points to the document
        with the highest score.
    """
    query = query.lower()
    query_tokenized = tokenizer.tokenize(query)

    results = []


    # make term counts for the query
    query_word_counts = {}

    for word in query_tokenized:
        if word in query_word_counts:
            query_word_counts[word] += 1
        else:
            query_word_counts[word] = 1

    # calculate the norm for q
    q_norms = 0

    for term in query_word_counts:
        counts = query_word_counts[term]
        if term in idf:
            q_norms += (counts * idf[term]) ** 2

    q_norms = np.sqrt(q_norms)


    # call score function
    scores = score_func(query_word_counts, index, idf)

    for doc in scores:
        results.append((scores[doc] / (q_norms * doc_norms[doc]), doc))


    # sort the tuples by score
    results.sort(key = lambda x: x[0], reverse=True)

    return results