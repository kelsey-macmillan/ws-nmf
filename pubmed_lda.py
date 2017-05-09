from src.model import lda
from src.features import tf
from src.utils.post_processing import *
from src.utils.pubmed_import import import_pubmed_data
import pprint
import pandas as pd
from collections import defaultdict


def main(docs, labels, all_labels):

    # create dictionaries {label: label_id} and {label_id: label}
    label_to_id_dict = {v: n for n, v in enumerate(all_labels)}
    id_to_label_dict = {v: k for k, v in label_to_id_dict.items()}

    # Filter out docs that are less than 250 characters
    docs_labels = filter(lambda x: len(x[0]) > 250, zip(docs, labels))
    docs = zip(*docs_labels)[0]
    labels = zip(*docs_labels)[1]

    # need to transform 'labels' from strings to indices
    label_ids = [[label_to_id_dict[label] for label in label_list] for label_list in labels]

    # Vectorize
    print 'vectorizing...'
    tf_vectorizer = tf.Vectorizer(vocab_size=2000)
    tf_vectorizer.fit(docs)
    doc_term_matrix, terms = tf_vectorizer.transform(docs)

    # Factorize (weakly supervised)
    print 'running lda....'
    lda_model = lda.Model(doc_term_matrix, K=len(all_labels))
    lda_model.train()
    doc_topic_matrix = lda_model.W
    topic_term_matrix = lda_model.H

    # Create useful dictionaries
    # {topic id: terms}
    topic_to_term_dict = create_topic_to_term_dict(topic_term_matrix, terms)
    # {doc id: [(topic id, normalized strength)]}
    doc_to_topic = create_doc_to_topic_dict(doc_topic_matrix, cutoff=0.001) # higher cutoff reduces dictionary size
    # {doc id: [(label id, normalized strength)]}
    doc_to_label = defaultdict(list)
    for doc_ind, label_list in enumerate(label_ids):
        for label in label_list:
            doc_to_label[doc_ind].append((label, 1))

    # Compute topic to label similarity matrix
    print 'computing similarity matrix...'
    similarity = compute_similarity_matrix(doc_to_topic, doc_to_label)

    # Run hungarian algorithm
    print 'running hungarian algorithm....'
    avg_score, sorted_matches, matched_similarity = match_similarity_matrix(similarity)

    # Print assignment score
    print 'Average similarity: %s' % avg_score

    # Print top 50 matched assignment
    matched_topic_terms = [(round(score, 3), id_to_label_dict[label_ind], topic_to_term_dict[topic_ind])
                              for score, topic_ind, label_ind in sorted_matches]
    pprint.pprint(matched_topic_terms[:50])

    # Determine number of "resolved" topics (similarity > 0.1)
    n_resolved = len([score for score, topic_ind, label_ind in sorted_matches if score > 0.1])

    # Print number of toipcs resolved
    print 'Numer of topics resolved: %s' % n_resolved

    return avg_score, n_resolved

if __name__ == "__main__":

    docs, labels, all_labels= import_pubmed_data('data/medline17n0001.xml')

    print 'running model...'
    avg_score, n_resolved = main(docs, labels, all_labels)
    print avg_score
    print n_resolved

    # Initialize results data frame
    df = pd.DataFrame()

    # Iterate
    for rep in range(3):

        # Run model
        avg_score, n_resolved = main(docs, labels, all_labels)

        # Add iteration to data frame
        data =   {'avg_similarity': avg_score,
                'n_topics_resolved': n_resolved,
                'rep': rep+1}
        df = df.append(data, ignore_index=True)

    df.to_csv('results/pubmed_lda.csv')
    print df