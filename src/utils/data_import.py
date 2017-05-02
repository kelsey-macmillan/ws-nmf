import csv
import numpy as np
from scipy.sparse import csr_matrix


def save_string_list_as_csv(l, filename):
    with open(filename, 'wb') as f:
        wr = csv.writer(f)
        wr.writerow([s.encode('utf-8') for s in l])


def save_int_list_as_csv(l, filename):
    with open(filename, 'wb') as f:
        wr = csv.writer(f)
        wr.writerow([n for n in l])


def save_matrix_as_csv(matrix, filename):
    np.savetxt(filename, matrix, delimiter=",")


def csv_to_matrix(filename):
    return np.genfromtxt(filename, delimiter=",")


def csv_to_list(filename):
    with open(filename, 'rb') as f:
        return csvfile_to_list(f)


def csvfile_to_list(f):
    l = []
    reader = csv.reader(f)
    for row in reader:
        l=l+row
    return l


def save_sparse_csr(array, filename):
    np.savez(filename, data=array.data, indices=array.indices, indptr=array.indptr, shape=array.shape)


def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])


