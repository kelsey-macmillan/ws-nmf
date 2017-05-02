import numpy as np
from scipy.sparse import csr_matrix
from src.utils.data_import import save_sparse_csr, load_sparse_csr


class Model:
    #TODO: add predict method

    def __init__(self, V, labels, K=None, train=True):
        """
        V: doc-term matrix (n docs x n terms)
        labels: list of lists, the order of the sublists should match the order of the docs,
                and each sublist is a list of topic id tags
        K: number of topics (n topics)
        """
        self.V = np.array(V) # coerce to array for correct multiplication type

        if train:
            self.labels = labels
            self.K = K
            self.L = self.create_constraint_matrix()
            self.H = self.initialize_H()
            self.W = self.initialize_W()
            self.E = self.create_weight_matrix()

    def create_constraint_matrix(self):
        """
        :param topic_labels: ordered by document order, elements are list of ints for labeled doc, and empty list for unlabeled doc
        :param K: total number of topics (known + unknown)
        :return:
        """
        L = np.ones((len(self.labels), self.K)) * 1  # initialize matrix of ones

        for document_index, topic_index_list in enumerate(self.labels):
            if len(topic_index_list) > 0:  # if document has been labeled
                L[document_index, :] = 0  # set all labels to zero for that doc initially

                for topic_index in topic_index_list:
                    L[document_index, topic_index] = 1  # set labeled topic / document to 1
        return L

    def initialize_H(self):
        # V is doc-term np matrix, K is number of topics
        # return initialized topic-term matrix

        H = np.empty((self.K, self.V.shape[1]))  # initialize empty matrix

        for k in range(self.K):

            # get any docs that are labeled with this topic
            topic_k_docs = [doc_ind for doc_ind, topics in enumerate(self.labels) if k in topics]

            # if any docs are found, use those to initialize
            if len(topic_k_docs) > 0:
                new_row = np.apply_along_axis(np.mean, 0, self.V[topic_k_docs, :])  # average them
                H[k, :] = new_row  # add to initialized matrix

            # other pick 20 random rows
            else:
                rand_docs = np.random.randint(low=0, high=self.V.shape[0], size=20)  # get 20 random rows
                new_row = np.apply_along_axis(np.mean, 0, self.V[rand_docs, :])  # average them
                H[k, :] = new_row  # add to initialized matrix
        return H

    def initialize_W(self):
        # V is doc-term matrix, K is number of topics
        # return initialized doc-topic matrix
        W = np.random.rand(self.V.shape[0], self.K) * np.array(self.L)  # random from Unif[0,1] o L
        return W

    def create_weight_matrix(self):
        """Function to create matrix to re-weight error function"""
        weights = np.ones(self.V.shape)

        n_labeled = sum([1 if len(label_list) > 0 else 0 for label_list in self.labels])

        for document_index, topic_index_list in enumerate(self.labels):
            if len(topic_index_list) > 0:  # if document has been labeled
                # weight the document more heavily (inverse freq)
                weights[document_index, :] = float(len(self.labels)) / n_labeled

        return weights

    def train(self, max_iter=100):
        """
        :param V: doc-term np matrix (rows are docs, columns are terms)
        :param K: number of topics
        :param labels: ordered by document order, elements are list of ints for labeled doc, and empty list for unlabeled doc
        :return: W (doc-topic) , H (topic-term)
        """
        e_prev = self.calculate_error() # calculate error

        for i in range(max_iter): # 100 is the max number of update iterations

            # update non-zero H
            mask = np.nonzero(self.H)

            H_update_numerator = np.array(np.dot((self.W * self.L).T, self.V * self.E)) # to array for element wise update
            H_update_denom = np.array(np.dot((self.W * self.L).T, np.dot((self.W*self.L), self.H) * self.E)) # to array for element wise update

            self.H[mask] = self.H[mask] * (H_update_numerator[mask] / (H_update_denom[mask]))

            # update non-zero W
            mask = np.nonzero(self.W)

            W_update_numerator = np.array(np.dot(self.V * self.E, self.H.T) * self.L) # to array for element wise update
            W_update_denom = np.array(np.dot(np.dot(self.W * self.L, self.H) * self.E, self.H.T) * self.L) # to array for element wise update

            self.W[mask] = self.W[mask] * (W_update_numerator[mask] / (W_update_denom[mask]))

            # recalculate error
            e_new = self.calculate_error()

            print 'Current loss is: ' + str(e_new)  # for debugging
            # check if error has converged (less than 0.01% change)
            if (e_prev - e_new) / e_prev < 0.0001:
                break
            else:
                e_prev = e_new

    def calculate_error(self):
        """Return MSE for model, squared loss of V-W*H."""
        sse = np.linalg.norm(self.V - np.matrix(self.W) * np.matrix(self.H), ord='fro')
        mse = sse / (self.V.shape[0] * self.V.shape[1])
        return mse

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


if __name__ == '__main__':

    tiny_V = np.array([[5, 3, 0, 1],
                        [4, 0, 0, 1],
                        [1, 1, 0, 5],
                        [1, 0, 0, 4],
                        [0, 1, 5, 4]])


    tiny_labels = [[0, 1],
                   [0],
                   [],
                   [],
                   []]

    K = 3

    ws_nmf = Model(tiny_V, tiny_labels, K)
    ws_nmf.train()
    tiny_W = ws_nmf.W
    tiny_H = ws_nmf.H

    np.set_printoptions(precision=3, suppress=True)
    print '\nW * H:'
    print np.dot(tiny_W, tiny_H)