import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from src.utils.data_import import csvfile_to_list
from src.utils.tokenizer import tokenize
from src.utils.text_preprocessing import clean_text
import pkg_resources


class Vectorizer:

    def __init__(self, vocab_size=None, train=True):

        if train:
            self.vocab_size = vocab_size
            self.get_tfidf_vectorizer()

    def fit(self, corpus):
        """corpus should be a list of strings, each corresponding to a doc"""
        self.vectorizer.fit(corpus)

    def transform(self, corpus):
        """corpus should be a list of strings, each corresponding to a doc"""
        doc_term_matrix = self.vectorizer.transform(corpus)
        doc_term_matrix = doc_term_matrix.todense()
        terms = self.vectorizer.get_feature_names()
        return doc_term_matrix, terms

    def get_tfidf_vectorizer(self):
        """Get TFIDF vectorizer model object from scikit."""

        # STOP WORDS
        with pkg_resources.resource_stream(__name__, 'GENERAL_STOPWORDS.txt') as general_f:
            general_stopwords = csvfile_to_list(general_f)

        tfidf_vectorizer = TfidfVectorizer(preprocessor=clean_text,
                                           tokenizer=tokenize,
                                           stop_words=general_stopwords,
                                           use_idf=True,
                                           ngram_range=(1, 1),
                                           norm='l2',  # normalize term frequency vectors (long docs repeat terms more)
                                           max_features=self.vocab_size,  # this should increase with the amount of data
                                           min_df=2)
        self.vectorizer = tfidf_vectorizer

    def load(self, filename):
        """Load vectorizer by unpickling."""
        with open(filename, 'rb') as fid:
            self.vectorizer = pickle.load(fid)

    def save(self, filename):
        """Save vectorizer by pickling."""
        with open(filename, 'wb') as fid:
            pickle.dump(self.vectorizer, fid)