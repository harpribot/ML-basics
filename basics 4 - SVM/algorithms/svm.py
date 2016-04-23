'''
    Author: Harshal Priyadarshi
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import normalize
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import *
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedShuffleSplit
from pca import PCA
from operator import itemgetter


class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


class SparseKernelMachines:
    def __init__(self, train_x, train_y, test_x, test_y):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x_hard = test_x[0:5000,:]
        self.test_y_hard = test_y[0:5000,:]
        self.test_x_easy = test_x[5000:10000,:]
        self.test_y_easy = test_y[5000:10000,:]
        self.normalize_input_data()
        self.use_fraction_of_train_for_fitting()

    def normalize_input_data(self):
        # Normalize the features
        self.train_x = normalize(self.train_x, axis=1)
        self.test_x_easy = normalize(self.test_x_easy, axis=1)
        self.test_x_hard = normalize(self.test_x_hard, axis=1)

    def use_fraction_of_train_for_fitting(self):
        # use only a part of training data for training
        self.train_x, _, self.train_y, _ = train_test_split(self.train_x, \
                                                self.train_y, \
                                                train_size= 0.15,
                                                random_state=49)


    def fit(self, C=None, gamma=None, kernel=None,decision_function_shape=None):
        if(C == None):
            C = [4,8,12,16]

        if(gamma == None):
            gamma = np.arange( 0.1, 5.0, 1 ).tolist(),

        if(kernel == None):
            kernel = ['linear', 'poly', 'rbf', 'sigmoid']

        if(decision_function_shape == None):
            parameters = {
                'C' : C,
                'kernel' : kernel,
                'gamma' : gamma,
                'class_weight' : ['balanced']
            }
        else:
            parameters = {
                'C' : C,
                'kernel' : kernel,
                'gamma' : gamma,
                'class_weight' : ['balanced'],
                'decision_function_shape' : decision_function_shape
            }



        clf = GridSearchCV(svm.SVC(), parameters, n_jobs=4,\
            cv=StratifiedKFold(np.ravel(self.train_y), n_folds=5, shuffle=True), \
            verbose=2, refit=True)
        # Train the model on a part of training data
        clf.fit(self.train_x, np.ravel(self.train_y))

        return clf

    def fit_predict_SVC(self,C=None, gamma=None, kernel=None, decision_function_shape=None):
        if(C == None):
            C = [4,8,12,16]

        if(gamma == None):
            gamma = np.arange( 0.1, 5.0, 1 ).tolist(),

        if(kernel == None):
            kernel = ['linear', 'poly', 'rbf', 'sigmoid']

        # fit the classifier - using cross validation
        clf = self.fit(C,gamma,kernel)
        # predict the training data
        self.predicted_train = clf.predict(self.train_x)
        # predict the easy test data
        self.predicted_test_easy = clf.predict(self.test_x_easy)
        # predict the hard test data
        self.predicted_test_hard = clf.predict(self.test_x_hard)
        # get the train and test (easy + hard) accuracy
        self.train_accuracy = self.get_accuracy(self.predicted_train, self.train_y)
        self.test_accuracy_easy = self.get_accuracy(self.predicted_test_easy, \
                                                    self.test_y_easy)
        self.test_accuracy_hard = self.get_accuracy(self.predicted_test_hard, \
                                                    self.test_y_hard)

    def get_accuracy(self, y_predict, y_true):
        return sum(np.ravel(y_predict) == np.ravel(y_true))/float(y_true.size)

    def display_accuracy(self):
        print "Train Accuracy:%f\n Test Accuracy(Easy):%f\nTest Accuracy(Hard):%f"\
                                %(self.train_accuracy, \
                                  self.test_accuracy_easy, \
                                  self.test_accuracy_hard)

    def do_heatmapping_gamma_C(self):
        self.C_range = np.logspace(-2, 5, 10)
        self.gamma_range = np.logspace(-5, 2, 10)
        param_grid = dict(gamma=self.gamma_range, C=self.C_range)
        cv = StratifiedShuffleSplit(np.ravel(self.train_y), n_iter=5, test_size=0.2, random_state=42)
        self.grid = GridSearchCV(svm.SVC(),n_jobs=4,verbose=2, param_grid=param_grid, cv=cv)
        self.grid.fit(self.train_x, np.ravel(self.train_y))

    def plot_heatmap(self):
        param = [x[0] for x in self.grid.grid_scores_]
        scores = [x[1] for x in self.grid.grid_scores_]
        scores = np.array(scores).reshape(len(self.C_range), len(self.gamma_range))
        plt.figure(figsize=(8, 6))
        plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
        plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
                   norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
        plt.xlabel('gamma')
        plt.ylabel('C')
        plt.colorbar()
        plt.xticks(np.arange(len(self.gamma_range)), self.gamma_range, rotation=45)
        plt.yticks(np.arange(len(self.C_range)), self.C_range)
        plt.title('Validation accuracy')


    def get_best_C_gamma(self):
        param = [x[0] for x in self.grid.grid_scores_]
        scores = [x[1] for x in self.grid.grid_scores_]
        ## Get best C and gamma
        best_index = max(enumerate(scores), key=itemgetter(1))[0]

        best_C = param[best_index]['C']
        best_gamma = param[best_index]['gamma']

        return best_C, best_gamma


    def get_accuracy_diff_kernels(self,grid):
        param = [x[0] for x in grid.grid_scores_]
        scores = [x[1] for x in grid.grid_scores_]

        kernel_acc_map = {}
        best_kernel = ''
        best_accuracy = 0
        for i in range(0, len(param)):
            kernel = param[i]['kernel']
            cv_accuracy = scores[i]
            if(cv_accuracy > best_accuracy):
                best_kernel = kernel
                best_accuracy = cv_accuracy

            kernel_acc_map[kernel] = cv_accuracy

        return kernel_acc_map, best_kernel

    def get_test_accuracies(self):
        return self.test_accuracy_easy, self.test_accuracy_hard

    def get_acc_for_decision_func(self, grid):
        param = [x[0] for x in grid.grid_scores_]
        scores = [x[1] for x in grid.grid_scores_]

        decision_acc_map = {}
        for i in range(0, len(param)):
            func = param[i]['decision_function_shape']
            cv_accuracy = scores[i]
            decision_acc_map[func] = cv_accuracy

        return decision_acc_map
