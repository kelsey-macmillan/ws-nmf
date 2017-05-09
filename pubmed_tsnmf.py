from src.model import ws_nmf
from src.features import tfidf
from src.utils.utils import flatten
from src.utils.pubmed_import import import_pubmed_data
from src.utils.post_processing import *
import pprint
import pandas as pd
from collections import defaultdict,  Counter
import random


def main(docs, labels, all_labels, supervision_rate):

    # create dictionaries {label: label_id} and {label_id: label}
    label_to_id_dict = {v: n for n, v in enumerate(all_labels)}
    id_to_label_dict = {v: k for k, v in label_to_id_dict.items()}

    # Filter out docs that are less than 250 characters
    docs_labels = filter(lambda x: len(x[0]) > 250, zip(docs, labels))
    docs = zip(*docs_labels)[0]
    labels = zip(*docs_labels)[1]


    # need to transform 'labels' from strings to indices
    label_ids = [[label_to_id_dict[label] for label in label_list] for label_list in labels]

    # subset list of doc labels at given rate
    label_ids_sub = [label_id_list if random.uniform(0, 1) < supervision_rate else [] for label_id_list in label_ids]

    # calculate how many topics are captured
    known_topic_counter = Counter()
    for label_id in flatten(label_ids_sub):
        known_topic_counter[label_id] += 1

    topic_coverage = float(len(known_topic_counter)) / len(all_labels) # percentage of topics covered
    print 'Topic coverage: %s' % topic_coverage

    # Vectorize
    print 'vectorizing...'
    tfidf_vectorizer = tfidf.Vectorizer(vocab_size=2000)
    tfidf_vectorizer.fit(docs)
    doc_term_matrix, terms = tfidf_vectorizer.transform(docs)

    # Factorize (weakly supervised)
    print 'running ts-nmf....'
    ws_nmf_model = ws_nmf.Model(doc_term_matrix, label_ids_sub, K=len(all_labels))
    ws_nmf_model.train(max_iter=30)
    doc_topic_matrix_ws = ws_nmf_model.W
    topic_term_matrix_ws = ws_nmf_model.H

    # Create useful dictionaries
    # {topic id: terms}
    topic_to_term_dict_ws = create_topic_to_term_dict(topic_term_matrix_ws, terms)
    # {doc id: [(topic id, strength)]}
    doc_to_topic_ws = create_doc_to_topic_dict(doc_topic_matrix_ws, cutoff=0.001) # higher cutoff reduces dictionary size
    # {doc id: [(label id, strength)]}
    doc_to_label = defaultdict(list)
    for doc_ind, label_list in enumerate(label_ids):
        for label in label_list:
            doc_to_label[doc_ind].append((label, 1))

    # Compute topic to label similarity matrix
    print 'computing similarity matrix...'
    similarity_ws = compute_similarity_matrix(doc_to_topic_ws, doc_to_label)

    # Run hungarian algorithm
    print 'running hungarian algorithm....'
    score_ws, sorted_matches_ws, matched_similarity_ws = match_similarity_matrix(similarity_ws)

    # Print assignment score
    print 'Average similarity: %s' % score_ws

    # Print top 50 matched assignment
    matched_topic_terms_ws = [(round(score, 3), id_to_label_dict[label_ind], topic_to_term_dict_ws[topic_ind])
                              for score, topic_ind, label_ind in sorted_matches_ws]
    pprint.pprint(matched_topic_terms_ws[:50])

    # Determine number of "resolved" topics (similarity > 0.1)
    n_resolved = len([score for score, topic_ind, label_ind in sorted_matches_ws if score > 0.1])

    # Print number of toipcs resolved
    print 'Numer of topics resolved: %s' % n_resolved

    return topic_coverage, score_ws, n_resolved


if __name__ == "__main__":

    docs, labels, all_labels= import_pubmed_data('data/medline17n0001.xml')

    # Initialize results data frame
    df = pd.DataFrame()

    # Iterate
    for supervision_rate in [0.01, 0.1, 0.2, 0.5, 0.8]:
        for rep in range(3):

            print(supervision_rate)

            # Run model
            k, score, n_resolved = main(docs, labels, all_labels, supervision_rate)

            # Add iteration to data frame
            data = {'supervision': supervision_rate,
                    'topic_coverage': k,
                    'avg_similarity': score,
                    'n_topics_resolved': n_resolved,
                    'rep': rep+1}
            df = df.append(data, ignore_index=True)

    df.to_csv('results/pubmed_tsnmf.csv')
    print df