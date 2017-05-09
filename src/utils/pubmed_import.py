import xml.etree.ElementTree as ET
from src.utils.utils import flatten
from collections import Counter


def import_pubmed_data(pubmed_filename):
    print 'importing raw data...'
    e = ET.parse(pubmed_filename).getroot()

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
    keywords_set = [kw for kw in keywords_counter.keys() if keywords_counter[kw] >= 50]

    for d in pubmed_dicts:
        d['keywords'] = filter(lambda x: x in keywords_set, d['keywords'])

    # Now filter out documents that have no remaining keywords
    pubmed_dicts = filter(lambda x: len(x['keywords']) > 0, pubmed_dicts)

    print 'N unique keywords: %s' % len(keywords_set)
    print 'N docs: %s' % len(pubmed_dicts)

    # Create list of docs
    docs = [d['abstract'] for d in pubmed_dicts]

    # Create list of labels lists
    labels = [d['keywords'] for d in pubmed_dicts]

    # Create list of all unique labels
    all_labels = list(keywords_set)

    return docs, labels, all_labels


# For debugging
if __name__ == "__main__":

    docs, labels, all_labels = import_pubmed_data('../../data/medline17n0001.xml')

    print all_labels