'''
Author: Harshal Priyadarshi - hp7325
Affiliation : University of Texas at Austin
Code Nature : Script to call the classes
Classes: algorithms/svm.py --> contains all the SVM related methods in a class
         algorithms/pca.py --> for projecting the data into lower dimension
'''
import numpy as np
import pandas as pd
from algorithms.pca import PCA
from algorithms.svm import SparseKernelMachines
import matplotlib.pyplot as plt


train_feature_file = 'data/train_x.csv'
train_label_file = 'data/train_y.csv'
test_feature_file = 'data/test_x.csv'
test_label_file = 'data/test_y.csv'


print 'Beginning First Experiment..... May take a while'
## Experiment 1 - Varying C and gamma
num_eigens = 50
train_fraction = 0.01
pca_handler = PCA(train_feature_file, train_label_file, \
                    test_feature_file, test_label_file, \
                        num_eigens, train_fraction)
# transform the data to the eigenspace
pca_handler.transform()
# obtain the transformed data
train_x, test_x, train_y, test_y = pca_handler.get_new_features_and_labels()
# initialize the svm model
svm_handler = SparseKernelMachines(train_x, train_y, test_x, test_y)
## Draw the heatmap for varying test accuracy for changing gamma and C
svm_handler.do_heatmapping_gamma_C()
svm_handler.plot_heatmap()
best_C, best_gamma = svm_handler.get_best_C_gamma()
print 'Best C: %f, Best Gamma: %f' %(best_C, best_gamma)

print 'Experiment 1 Concluded. Moving Forward...'


print 'Beginning Second Experiment....'
## Experiment 2 - Vary the kernel and use the best (C,gamma) from previous experiments
svm_handler = SparseKernelMachines(train_x, train_y, test_x, test_y)
kernel = ['linear', 'poly', 'rbf', 'sigmoid']
grid = svm_handler.fit([best_C],[best_gamma], kernel)
kernel_dict, best_kernel = svm_handler.get_accuracy_diff_kernels(grid)
print 'RBF Accuracy --> Cross Validation Accuracy:%f' %(kernel_dict['rbf'])
print 'Sigmoid Accuracy --> Cross Validation Accuracy:%f' %(kernel_dict['sigmoid'])
print 'Polynomial Kernel Accuracy --> Cross Validation Accuracy:%f' %(kernel_dict['poly'])
print 'Linear Kernel Accuracy --> Cross Validation Accuracy:%f' %(kernel_dict['linear'])

print 'Experiment 2 Concluded. Moving Forward...'


print 'Beginning Third Experiment....This may take a while'
## Experiment 3 - Varying eigenspace dimension
eigen_range = range(1,100,10)
train_fraction = 0.01
easy_acc = []
hard_acc = []
# Do the SVM training for each of the eigenspace projection
for num_eigen in eigen_range:
    pca_handler = PCA(train_feature_file, train_label_file, \
                        test_feature_file, test_label_file, \
                            num_eigen, train_fraction)
    pca_handler.transform()
    train_x, test_x, train_y, test_y = pca_handler.get_new_features_and_labels()
    svm_handler = SparseKernelMachines(train_x, train_y, test_x, test_y)
    svm_handler.fit_predict_SVC([best_C],[best_gamma],[best_kernel])
    easy,hard = svm_handler.get_test_accuracies()
    easy_acc.append(easy)
    hard_acc.append(hard)

# Plot the test accyracy for each of the eigenspace dimensions
plt.figure()
plt.plot(eigen_range, easy_acc, label = 'Test:Easy')
plt.plot(eigen_range, hard_acc, label= 'Test:Hard')
plt.title('Varying Eigenspace Dimension')
plt.xlabel('Number of top eigenvectors')
plt.ylabel('Test Set Accuracy')
plt.legend()
print 'Experiment 3 Concluded. Moving Forward...'


print 'Beginning Fourth Experiment....'
## Experiment 4 - OVO vs OVR which is better ?
decision_function_shape = ['ovo', 'ovr']
num_eigen = 50
pca_handler = PCA(train_feature_file, train_label_file, \
                    test_feature_file, test_label_file, \
                        num_eigen, train_fraction)
pca_handler.transform()
train_x, test_x, train_y, test_y = pca_handler.get_new_features_and_labels()
svm_handler = SparseKernelMachines(train_x, train_y, test_x, test_y)
clf = svm_handler.fit([best_C], [best_gamma], [best_kernel], decision_function_shape)
decision_acc_map = svm_handler.get_acc_for_decision_func(clf)
print 'One v/s One Model Accuracy --> Cross Validation Accuracy:%f' %(decision_acc_map['ovo'])
print 'One v/s Rest Model Accuracy --> Cross Validation Accuracy:%f' %(decision_acc_map['ovr'])



print 'Beginning Last Experiment.... May take a while'
## Experiment 5 - Best Overall
# PART 1  - Reduce the feature dimension to the eigenspace of top K eigenvectors
num_eigens = 50
train_fraction = 0.01
pca_handler = PCA(train_feature_file, train_label_file, \
                    test_feature_file, test_label_file, \
                        num_eigens, train_fraction)
# transform the features to the space of the best k (= num_eigens) eigenvector
pca_handler.transform()
# get the new transformed data
train_x, test_x, train_y, test_y = pca_handler.get_new_features_and_labels()
# PART 2 - Run SVM on the new data to check the accuracy
# initialize the svm handler
svm_handler = SparseKernelMachines(train_x, train_y, test_x, test_y)
# train and test on the SVC model
C = [4,8,12,16]
gamma = np.arange( 1, 50, 10 )/float(10)
kernel = ['linear', 'poly', 'rbf', 'sigmoid']
svm_handler.fit_predict_SVC(C,gamma,kernel)
# display the result of training

svm_handler.display_accuracy()
print 'Last Experiment Concluded'
plt.show()
print 'All done.. See your plots'
print 'Exited Successfully, with 0 errors'

'''
BEST RESULTS Obtained at the end of the experiment
Train Accuracy:1.000000
Test Accuracy(Easy):0.986200
Test Accuracy(Hard):0.963400
'''
