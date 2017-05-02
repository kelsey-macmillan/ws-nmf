import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import scipy as sp


def create_topic_to_term_dict(topic_term_matrix, terms):
    """
    :param topic_term_matrix: the topic-term matrix from NMF factorization
    :param terms: the list of terms that correspond to the column indices in the topic_term_matrix
    :return: a dictionary {topic_index: "terms"}
    """

    topic_dict = {}  # initialize dictionary

    for topic_ind in range(topic_term_matrix.shape[0]):

        # get row in matrix
        topic_row = topic_term_matrix[topic_ind, :]

        # get terms that are at least 10% as strong as the strongest term in each topic
        sorted_terms = sorted(zip(topic_row, terms), reverse=True)
        max_strength = sorted_terms[0][0]
        top_terms = [term for strength, term in sorted_terms if strength > max_strength*0.10]

        # add to dictionary
        topic_dict[topic_ind] = ','.join(top_terms[:3]) # limit to four terms

    return topic_dict


def create_doc_to_topic_dict(doc_topic_matrix, cutoff=1e-6):
    """
    :param doc_topic_matrix: the document-topic matrix from NMF factorization
    :param cutoff: topic strength cutoff to include topic in document (1e-6 is low default)
    :return: dictionary {doc_id: [(topic_id, strength)]}
    """

    # first, normalize doc_topic matrix
    doc_topic_matrix /= np.max(doc_topic_matrix)

    doc_dict = defaultdict(dict)  # initialize dictionary

    for doc_ind in range(doc_topic_matrix.shape[0]):

        # get row in matrix
        doc_row = doc_topic_matrix[doc_ind, :]

        # get topic indices and strength with strength above cutoff
        topic_indices = np.array(range(doc_topic_matrix.shape[1]))[doc_row > cutoff]
        topic_strengths = doc_row[doc_row > cutoff]

        # add to dictionary
        doc_dict[doc_ind] = zip(topic_indices, topic_strengths)

    return doc_dict


def match_similarity_matrix(sim):
    # use hungarian algorithm to find optimal mapping
    ind_1, ind_2 = sp.optimize.linear_sum_assignment(-sim)

    # calculate average similarity of mapped indices
    average_sim_score = np.mean(sim[ind_1, ind_2])

    # sort matches by similarity score
    sorted_matches = sorted(zip(sim[ind_1, ind_2], ind_1, ind_2), reverse=True)

    # create mapped and sorted similarity matrix
    matched_similarity = [[sim[sorted_matches[r_ind][1], sorted_matches[c_ind][2]]
                           for c_ind in range(len(ind_2))] for r_ind in range(len(ind_1))]

    return average_sim_score, sorted_matches, matched_similarity


def compute_similarity_matrix(doc_to_topic_1, doc_to_topic_2):
    """
    :param doc_to_topic_1: dictionary {doc_id: [(topic_id, strength)]}
    :param doc_to_topic_2: dictionary {doc_id: [(topic_id, strength)]}
    :return: similarity matrix with topic_1 indices as rows, and topic_2 indices as cols
    """

    # reverse dictionaries to be {topic_id: [(doc_id, strength)]
    topic_to_doc_1 = reverse_doc_to_topic_dict(doc_to_topic_1)
    topic_to_doc_2 = reverse_doc_to_topic_dict(doc_to_topic_2)

    similarity = np.empty((len(topic_to_doc_1), len(topic_to_doc_2)))
    for topic_id_1, doc_list_1 in topic_to_doc_1.items():
        for topic_id_2, doc_list_2 in topic_to_doc_2.items():
            similarity[topic_id_1, topic_id_2] = weighted_jaccard_distance(doc_list_1, doc_list_2)

    return similarity


def reverse_doc_to_topic_dict(doc_to_topic):
    topic_to_doc = defaultdict(list)
    for doc_ind, topic_list in doc_to_topic.items():
        for topic_id, topic_strength in topic_list:
            topic_to_doc[topic_id].append((doc_ind, topic_strength))
    return topic_to_doc


def weighted_jaccard_distance(list_1, list_2):
    """
    :param list_1: list of (id, weight) tuples
    :param list_2: list of (id, weight) tuples
    :return: weighted similarity score
    reference - http://static.googleusercontent.com/media/research.google.com/en/us/pubs/archive/36928.pdf
    """
    # convert id lists to sets
    id_set_1 = set(zip(*list_1)[0])
    id_set_2 = set(zip(*list_2)[0])

    # determine intersection and outside of intersection
    both_sets_ids = id_set_1 & id_set_2
    set_1_ids = id_set_1 - (id_set_1 & id_set_2)
    set_2_ids = id_set_2 - (id_set_1 & id_set_2)

    # convert lists of (id, weight) tuples to {id: weight}
    dict_1 = dict(list_1)
    dict_2 = dict(list_2)

    numerator = 0.0
    denominator = 0.0

    for k in both_sets_ids:
        numerator += min(dict_1[k], dict_2[k])
        denominator += max(dict_1[k], dict_2[k])
    for k in set_1_ids:
        denominator += dict_1[k]
    for k in set_2_ids:
        denominator += dict_2[k]

    return numerator/denominator


def l1_similarity(list_1, list_2):
    """
    :param list_1: list of (id, weight) tuples
    :param list_2: list of (id, weight) tuples
    :return: l1 similarity = 1 - l1_norm(list_1 - list_2)
    """

    # get normalization constants for each list
    norm_constant_1 = sum(zip(*list_1)[1])
    norm_constant_2 = sum(zip(*list_2)[1])

    # convert id lists to sets
    id_set_1 = set(zip(*list_1)[0])
    id_set_2 = set(zip(*list_2)[0])

    # determine intersection and outside of intersection
    both_sets_ids = id_set_1 & id_set_2
    set_1_ids = id_set_1 - (id_set_1 & id_set_2)
    set_2_ids = id_set_2 - (id_set_1 & id_set_2)

    # convert lists of (id, weight) tuples to {id: weight}
    dict_1 = dict(list_1)
    dict_2 = dict(list_2)

    l1_dist = 0

    for k in both_sets_ids:
        l1_dist += abs(dict_1[k]/norm_constant_1 - dict_2[k]/norm_constant_2)
    for k in set_1_ids:
        l1_dist += abs(dict_1[k]/norm_constant_1)
    for k in set_2_ids:
        l1_dist += abs(dict_2[k]/norm_constant_2)

    # return 1 minus distance to get similarity
    return 1.0 - l1_dist


def plot_maximal_strengths(strength_matrix, fname=None):
    maximal_strengths = np.apply_along_axis(np.max, 1, strength_matrix)
    fig = plt.figure()
    plt.hist(np.log10(maximal_strengths[maximal_strengths != 0.0].flatten()))
    plt.show()
    if fname:
        plt.savefig(fname)


def plot_strengths(strength_matrix, fname=None):
    fig = plt.figure()
    plt.hist(np.log10(strength_matrix[(strength_matrix != 0.0) & (strength_matrix > 1e-12)].flatten()))
    plt.show()
    if fname:
        plt.savefig(fname)





