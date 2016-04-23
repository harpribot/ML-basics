'''
    Author: Harshal Priyadarshi
'''

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import normalize
from scipy import linalg


class PCA:
    def __init__(self,train_file, train_label_file, test_file, test_label_file,\
                    num_eigens,train_fraction):
        self.train_file = train_file
        self.train_label_file = train_label_file
        self.test_file = test_file
        self.test_label_file = test_label_file
        self.num_eigens = num_eigens
        self.train_fraction = train_fraction

    def transform(self):
        self.load_data()
        self.chooseTrainingData()
        self.findEigenDigits()
        self.project_data()

    def load_data(self):
        self.X_train = pd.read_csv(self.train_file, header=None).values
        self.y_train = pd.read_csv(self.train_label_file, header=None).values
        self.X_test = pd.read_csv(self.test_file,header=None).values
        self.y_test = pd.read_csv(self.test_label_file, header=None).values

    def chooseTrainingData(self):
        self.A_t, _, _, _ = train_test_split(self.X_train, self.y_train, \
                                            train_size=self.train_fraction, random_state=49)
        self.A = self.A_t.T


    def findEigenDigits(self):
        # Find the mean image
        self.M = np.mean(self.A,axis=1)
        self.M = self.M[:,np.newaxis]
        # Subtract the mean from the images
        self.A = self.A - self.M
        # Find the covariance matrix
        eigval,eigvec = linalg.eig(np.dot(self.A.T, self.A))

        # Sort the eigenvectors according to decreasing order sorted eigenvalues for A' * A
        idx = eigval.argsort()[::-1]
        eigval = eigval[idx]
        eigvec = eigvec[:,idx]

        # top k (= num_eigens) eigenvectors for A * A'
        eigvec = np.dot(self.A ,eigvec)
        eigvec = eigvec[:,0:self.num_eigens]

        # take only the real part of the eigenvectors and normalize it
        self.eigvec = normalize(eigvec.real,axis=0) # 784 x 600


    def project_data(self):
        self.X_train = self.X_train - self.M.T
        self.X_test = self.X_test - self.M.T
        self.X_train = np.dot(self.X_train, self.eigvec)
        self.X_test = np.dot(self.X_test, self.eigvec)

    def get_new_features_and_labels(self):
        return self.X_train, self.X_test, self.y_train, self.y_test
