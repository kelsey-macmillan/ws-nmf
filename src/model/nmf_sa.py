import numpy as np
from scipy.sparse import csr_matrix
from src.utils.data_import import save_sparse_csr, load_sparse_csr
import random
import copy

class Model:

    def __init__(self, V, K=None, train=True):
        """
        V: doc-term matrix (n docs x n terms)
        K: number of topics (n topics)
        """
        self.V = np.array(V)  # coerce to array for correct multiplication type

        if train:
            self.K = K
            self.H = self.initialize_H()
            self.W = self.initialize_W()
            self.W_alt = copy.deepcopy(self.W)

    def initialize_H(self):
        """
        Initialize topic-term matrix using random V-col initialization.
        :param V: doc-term matrix (n docs x n terms)
        :param K: number of topics
        :return: initialized topic-term matrix, H (n topics x n terms)
        """
        H = np.empty((self.K, self.V.shape[1]))  # initialize empty matrix
        for k in range(self.K):
            rand_row_i = np.random.randint(low=0, high=self.V.shape[0], size=20)  # get 20 random rows
            new_row = np.apply_along_axis(np.mean, 0, self.V[rand_row_i, :])  # average them
            H[k, :] = new_row  # add to initialized matrix
        return H

    def initialize_W(self):
        """
        Initialize doc-topic matrix using random uniform initialization.
        :param V: doc-term matrix (n docs x n terms)
        :param K: number of topics (n topics)
        :return: initialized doc-topic matrix, W (n docs x n topics)
        """
        W = np.random.rand(self.V.shape[0], self.K)  # random from Unif[0,1]
        return W

    def train(self, T0=50, max_iter=300):
        """
        V: doc-term matrix (n docs x n terms)
        W,H: factorization W*H, W is doc-topic, H is topic-term
        """

        ct = 0  # initialize counter for quitting condition

        for i in range(max_iter):

            # Update temperature
            T = T0*0.95 ** i

            # Update H and W
            self.update_H()
            self.update_W()

            # Alter W of alt state by scrambling 10% of cols from current state
            self.W_alt = copy.deepcopy(self.W)
            rand_cols = np.random.choice(range(self.K), self.K / 10)
            self.W_alt[:, rand_cols] = np.random.rand(self.V.shape[0], len(rand_cols))

            # Calculate energies
            energy = self.calculate_error(alt=False)
            print energy
            energy_alt = self.calculate_error(alt=True)
            print energy_alt

            # Calculate transition probability
            if energy_alt < energy:
                prob = 1
            else:
                prob = np.exp(-(np.log(energy_alt) - np.log(energy)) / T)

            # Decide whether to transition state
            print 'Transfer probability is: ' + str(prob)
            if prob > random.uniform(0, 1):
                self.W = copy.deepcopy(self.W_alt)

            # Decide whether to quit
            if prob < 0.01:
                ct += 1
                if ct > 10:
                    break

    def calculate_error(self, alt=False):
        """Return MSE for model, squared loss of V-W*H. NOT weighted loss!"""
        if alt:
            sse = np.linalg.norm(self.V - np.matrix(self.W_alt) * np.matrix(self.H), ord='fro')
        else:
            sse = np.linalg.norm(self.V - np.matrix(self.W) * np.matrix(self.H), ord='fro')
        mse = sse / (self.V.shape[0] * self.V.shape[1])
        return mse

    def update_H(self):
        """Update H using multiplicative update rule from Lee and Seung."""
        mask = np.nonzero(self.H)  # update non-zero H only (avoid divide by zero error)
        numerator = np.array(np.dot(self.W.T, self.V))  # to array for element-wise update
        denominator = np.array(np.dot(self.W.T, np.dot(self.W, self.H)))  # to array for element-wise update
        self.H[mask] = self.H[mask] * (numerator[mask] / denominator[mask])

    def update_W(self):
        """Update W using multiplicative update rule from Lee and Seung."""
        mask = np.nonzero(self.W)  # update non-zero W
        numerator = np.array(np.dot(self.V, self.H.T))
        denominator = np.array(np.dot(self.W, np.dot(self.H, self.H.T)))
        self.W[mask] = self.W[mask] * (numerator[mask] / denominator[mask])

    def predict(self, max_iter=100):
        """
        V: the new document-term matrix using the same doc vectorizer that fit H (n docs x n terms)
        H: the fitted topic-term matrix from training phase (n topics x n terms)
        W: a document-topic matrix, W (n docs x n topics)
        """
        self.W = self.initialize_W()

        e_prev = self.calculate_error()

        for i in range(max_iter):  # 100 is the max number of update iterations

            self.update_W()

            # recalculate error
            e_new = self.calculate_error()

            print 'Current loss is: ' + str(e_new)  # for debugging
            # check if error has converged (less than 0.01% change)
            if (e_prev - e_new) / e_prev < 0.0001:
                break
            else:
                e_prev = e_new

    def save(self, dir, model_name):
        """
        Save model by saving sparse factorized matrices.
        :param dir: directory
        :param model_name: model_name to be used in file naming
        """

        # force bottom 80% of values in matrices to zero
        cutoff_W = np.percentile(self.W.flatten(), 0.8)
        self.W[self.W < cutoff_W] = 0

        cutoff_H = np.percentile(self.H.flatten(), 0.8)
        self.H[self.H < cutoff_H] = 0

        # save result as csr_matrix
        save_sparse_csr(csr_matrix(self.W), dir + "/" + model_name + '_W.npz')
        save_sparse_csr(csr_matrix(self.H), dir + "/" + model_name + '_H.npz')

    def load(self, dir, model_name):
        """Load model by loading factorized matrices. Returns dense matrix."""
        self.H = load_sparse_csr(dir + "/" + model_name + '_H.npz')
        self.H = self.H.toarray()
        self.W = load_sparse_csr(dir + "/" + model_name + '_W.npz')
        self.W = self.W.toarray()
        self.K = self.H.shape[0]

