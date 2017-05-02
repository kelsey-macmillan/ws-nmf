import pickle
from sklearn.decomposition import LatentDirichletAllocation

# TODO: make this work more like the nmf model
class Model:

    def __init__(self, V, K=None, train=True):
        """
        V: doc-term matrix (n docs x n terms)
        K: number of topics (n topics)
        """
        self.V = V
        if train:
            self.K = K
            self.model = LatentDirichletAllocation(n_topics=self.K, max_iter=25, learning_method='batch')

    def train(self):
        """
        V: doc-term matrix (n docs x n terms)
        W,H: factorization W*H, W is doc-topic, H is topic-term
        """
        self.W = self.model.fit_transform(self.V)
        self.H = self.model.components_

    def predict(self):
        """
        :return: doc-topic matrix (W), where V = W*H
        """
        self.W = self.model.transform(self.V)

    def load(self,filename):
        """Load vectorizer by unpickling."""
        with open(filename, 'rb') as fid:
            self.model = pickle.load(fid)
        self.H = self.model.components_

    def save(self, filename):
        """Save vectorizer by pickling."""
        with open(filename, 'wb') as fid:
            pickle.dump(self.model, fid)

    def calculate_perplexity(self):
        """
        :return: perplexity of model for this dataset
        """
        return self.model.perplexity(self.V, self.W)