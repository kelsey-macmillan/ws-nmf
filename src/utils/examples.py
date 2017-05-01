

def print_examples(doc_ids, docs, doc_to_label, doc_to_topic, topic_to_term_dict, id_to_label_dict):
    """
    :param doc_ids: list of doc ids to print label and topic info for
    :param docs: list of all docs
    :param doc_to_label: {doc id: [(label id, normalized strength)]}
    :param doc_to_topic: {doc id: [(topic id, normalized strength)]}
    :param topic_to_term_dict: {topic id: terms}
    :param id_to_label_dict: {label id: term}
    :return:
    """

    # for each doc
    for doc_id in doc_ids:

        # Print the doc id
        print doc_id

        # Print the doc
        print docs[doc_id]

        # Print the labels
        print 'Labels:'
        for label_id, _ in doc_to_label[doc_id]:
            print id_to_label_dict[label_id]

        # Print the topics
        print 'Topics:'
        max_topic_strength = max(zip(*doc_to_topic[doc_id])[1])
        for topic_id, strength in doc_to_topic[doc_id]:
            if strength > max_topic_strength*0.5:
                print (topic_to_term_dict[topic_id], strength)