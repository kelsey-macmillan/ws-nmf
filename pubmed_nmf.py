from src.model import nmf
from src.features import tfidf
from src.utils.post_processing import *
from src.utils.utils import flatten
import pprint
import pandas as pd
from collections import defaultdict
import xml.etree.ElementTree as ET
from collections import defaultdict, Counter


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

    return avg_score, n_resolved

if __name__ == "__main__":

    ########### IMPORT DATA ############
    print 'importing raw data...'
    e = ET.parse('data/medsample1.xml').getroot()

    # Initialize data dictionaries
    pubmed_dicts = []

    print 'parsing XML..'
    for article in e.findall('PubmedArticle'):

        # Get article ID
        article_id = article.find(".//ArticleId[@IdType='pubmed']").text

        # Get abstract text (pass if no text)
        find_abstracts = article.findall(".//AbstractText")
        if len(find_abstracts) > 0:
            cur_abstract = ' '.join([abstract.text for abstract in find_abstracts])
        else:
            continue

        # Get keywords (pass if no keywords)
        find_keywords = article.findall(".//MeshHeading/DescriptorName")
        if len(find_keywords) > 0:
            cur_keywords = [keyword.text for keyword in find_keywords]
        else:
            continue

        pubmed_dicts.append({'article_id': article_id,
                             'abstract': cur_abstract,
                             'keywords': cur_keywords})

    print 'filtering infrequent keywords...'
    # Create counter with keywords
    keywords_counter = Counter()
    for kw in flatten([d['keywords'] for d in pubmed_dicts]):
        keywords_counter[kw] += 1

    # Filter out keywords that occur less than 5 times
    keywords_set = [kw for kw in keywords_counter.keys() if keywords_counter[kw] >= 5]

    for d in pubmed_dicts:
        d['keywords'] = filter(lambda x: x in keywords_set, d['keywords'])

    # Now filter out documents that have no remaining keywords
    pubmed_dicts = filter(lambda x: len(x['keywords']) > 0, pubmed_dicts)

    print 'N unique keywords: %s'  % len(keywords_set)
    print 'N docs: %s' % len(pubmed_dicts)

    # Create list of docs
    docs = [d['abstract'] for d in pubmed_dicts]

    # Create list of labels lists
    labels = [d['keywords'] for d in pubmed_dicts]

    # Create list of all unique labels
    all_labels = list(keywords_set)


    ########### RUN MODEL ############
    print 'running model...'
    avg_score, n_resolved = main(docs, labels, all_labels)
    print avg_score
    print n_resolved


    # # Initialize results data frame
    # df = pd.DataFrame()
    #
    # # Iterate
    # for rep in range(30):
    #
    #     # Run model
    #     avg_score, n_resolved = main(docs, labels)
    #
    #     # Add iteration to data frame
    #     data =   {'avg_similarity': avg_score,
    #             'n_topics_resolved': n_resolved,
    #             'rep': rep+1}
    #     df = df.append(data, ignore_index=True)
    #
    # df.to_csv('results/reuters_lda.csv')
    # print df
