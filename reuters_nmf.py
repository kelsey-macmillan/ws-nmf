from src.model import nmf
from src.features import tfidf
from src.utils.post_processing import *
import pprint
import pandas as pd
from nltk.corpus import reuters
from collections import defaultdict
from src.utils.examples import print_examples


def main(docs, labels):

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
    tfidf_vectorizer = tfidf.Vectorizer(vocab_size=2000)
    tfidf_vectorizer.fit(docs)
    doc_term_matrix, terms = tfidf_vectorizer.transform(docs)

    # Factorize (weakly supervised)
    nmf_model = nmf.Model(doc_term_matrix, K=len(all_labels))
    nmf_model.train(max_iter=30)
    doc_topic_matrix = nmf_model.W
    topic_term_matrix = nmf_model.H

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
    similarity = compute_similarity_matrix(doc_to_topic, doc_to_label)

    # Run hungarian algorithm
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

    # Print examples of documents
    doc_ids = [29, 38, 13, 28, 41]
    #print_examples(doc_ids, docs, doc_to_label, doc_to_topic, topic_to_term_dict, id_to_label_dict)

    return avg_score, n_resolved

if __name__ == "__main__":

    # Import data
    file_ids = reuters.fileids()
    all_labels = reuters.categories()
    docs = [' '.join(reuters.words(file_id)) for file_id in file_ids]
    labels = [reuters.categories(file_id) for file_id in file_ids]

    # Initialize results data frame
    df = pd.DataFrame()

    # Iterate
    for rep in range(30):

        # Run model
        avg_score, n_resolved = main(docs, labels)

        # Add iteration to data frame
        data = {'avg_similarity': avg_score,
                'n_topics_resolved': n_resolved,
                'rep': rep+1}
        df = df.append(data, ignore_index=True)

    df.to_csv('results/reuters_nmf.csv')
    print df
