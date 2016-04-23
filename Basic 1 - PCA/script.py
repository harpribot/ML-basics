import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from scipy import linalg



def load_data():
    X_train = pd.read_csv('data/train_x.csv', header=None).values
    y_train = pd.read_csv('data/train_y.csv', header=None).values
    X_test = pd.read_csv('data/test_x.csv',header=None).values
    y_test = pd.read_csv('data/test_y.csv', header=None).values

    return (X_train,X_test, y_train,y_test)


def chooseTrainingData(X_train, y_train, t):
    _, A_t, _,_ = train_test_split(X_train, y_train, test_size=t, random_state=49)
    A = A_t.T
    return A,A_t


def findEigenDigits(A):
    # Find the mean image
    M = np.mean(A,axis=1)
    M = M[:,np.newaxis]
    # Subtract the mean from the images
    A = A - M
    # Find the covariance matrix
    eigval,eigvec = linalg.eig(np.dot(A.T, A))

    # Sort the eigenvectors according to decreasing order sorted eigenvalues for A' * A
    idx = eigval.argsort()[::-1]
    eigval = eigval[idx]
    eigvec = eigvec[:,idx]

    # top k eigenvectors for A * A'
    eigvec = np.dot(A ,eigvec)

    # take only the real part of the eigenvectors and normalize it
    eigvec_real_normalized = normalize(eigvec.real,axis=0) # 784 x 600

    return (eigvec_real_normalized, M)




def plot_image(heading, images, num_cols,num_rows):
    image_shape = (28,28)
    plt.figure(figsize=(2. * num_cols, 2.26 * num_rows))
    plt.suptitle(heading, size=16)
    for i, digit in enumerate(images):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(digit.reshape(image_shape),cmap=plt.cm.gray)
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)


def reconstruction(image_sample, eigvec, mean, eig_num):
    image_sample = image_sample - mean.T
    projection = np.dot(image_sample,eigvec[:,:eig_num])
    image_reconstructed = np.dot(eigvec[:,:eig_num], projection.T) + mean

    return image_reconstructed




def project_data(X_train, X_test, eigvec, mean):
    X_train = X_train - mean.T
    X_test = X_test - mean.T
    X_train_projected = np.dot(X_train, eigvec)
    X_test_projected = np.dot(X_test, eigvec)

    return X_train_projected, X_test_projected



def KNeighborsClassification(X_train, y_train, X_test_easy, y_test_easy,\
        X_test_hard, y_test_hard, neighbors):
    ''' '''
    classifier = KNeighborsClassifier(n_neighbors = neighbors)
    classifier.fit(X_train, np.ravel(y_train))
    prediction_easy = classifier.predict(X_test_easy)
    prediction_hard = classifier.predict(X_test_hard)
    easy_accuracy = float(sum(np.ravel(prediction_easy) == \
            np.ravel(y_test_easy)))/y_test_easy.size
    hard_accuracy = float(sum(np.ravel(prediction_hard) == \
            np.ravel(y_test_hard)))/y_test_hard.size

    print "Easy Test Accuracy(KNN): " ,easy_accuracy
    print "Hard Test Accuracy(KNN): ", hard_accuracy
    return (prediction_easy,prediction_hard, easy_accuracy,hard_accuracy)

def LogisticClassification(X_train, y_train, X_test_easy, y_test_easy,X_test_hard, y_test_hard):
    classifier = LogisticRegression()
    classifier.fit(X_train, np.ravel(y_train))
    prediction_easy = classifier.predict(X_test_easy)
    prediction_hard = classifier.predict(X_test_hard)
    easy_accuracy = float(sum(np.ravel(prediction_easy) == \
            np.ravel(y_test_easy)))/y_test_easy.size
    hard_accuracy = float(sum(np.ravel(prediction_hard) == \
            np.ravel(y_test_hard)))/y_test_hard.size
    print "Easy Test Accuracy(Logistic REgression): " ,easy_accuracy
    print "Hard Test Accuracy(Logistic Regression): ", hard_accuracy
    return (prediction_easy,prediction_hard, easy_accuracy,hard_accuracy)

def ForestClassification(X_train, y_train, X_test_easy, y_test_easy,X_test_hard, y_test_hard, num_trees = 500):
    # normalize training and test data
    X_train = normalize(X_train, axis=1)
    X_test_easy = normalize(X_test_easy, axis=1)
    X_test_hard = normalize(X_test_hard, axis=1)
    classifier = RandomForestClassifier(n_estimators=num_trees)
    classifier.fit(X_train, np.ravel(y_train))
    prediction_easy = classifier.predict(X_test_easy)
    prediction_hard = classifier.predict(X_test_hard)
    easy_accuracy = float(sum(np.ravel(prediction_easy) == \
            np.ravel(y_test_easy)))/y_test_easy.size
    hard_accuracy = float(sum(np.ravel(prediction_hard) == \
            np.ravel(y_test_hard)))/y_test_hard.size
    print "Easy Test Accuracy(Random Forest): " ,easy_accuracy
    print "Hard Test Accuracy(Random Forest): ", hard_accuracy
    return (prediction_easy,prediction_hard, easy_accuracy,hard_accuracy)


def SvmClassification(X_train, y_train, X_test_easy, y_test_easy,X_test_hard, y_test_hard):
    # normalize training and test data
    X_train = normalize(X_train, axis=1)
    X_test_easy = normalize(X_test_easy, axis=1)
    X_test_hard = normalize(X_test_hard, axis=1)
    classifier = SVC(decision_function_shape='ovo',degree=3,C=15, gamma=0.1)
    classifier.fit(X_train, np.ravel(y_train))
    prediction_easy = classifier.predict(X_test_easy)
    prediction_hard = classifier.predict(X_test_hard)
    easy_accuracy = float(sum(np.ravel(prediction_easy) == \
            np.ravel(y_test_easy)))/y_test_easy.size
    hard_accuracy = float(sum(np.ravel(prediction_hard) == \
            np.ravel(y_test_hard)))/y_test_hard.size
    print "Easy Test Accuracy(SVC-rbf): " ,easy_accuracy
    print "Hard Test Accuracy(SVC-rbf): ", hard_accuracy
    return (prediction_easy,prediction_hard, easy_accuracy,hard_accuracy)

################################# Run Script ##################################
'''
num_eigens = np.array([5,10,20,50,100,200,300,400,500,600])
train_sz = np.array([20,40,80,100,150,200,300,400,500,600,700])
neighbors_val = np.array([1,2,3,4,5,6,7,8,9,10])
'''
# Unomment these to get the accuracy vector necessary for the plot
# Make sure to also substitute the accuracy_XXX_easy  with accuracy_XXX[0,0::] in training script for each function
# Make sure to also substitute the accuracy_XXX_hard  with accuracy_XXX[1,0::] in training script for each function
'''
accuracy_KNN = np.zeros((2,10));
accuracy_SVM = np.zeros((2,10));
accuracy_Forest = np.zeros((2,10));
accuracy_Logic = np.zeros((2,10));
'''
#START
print "Loading the Training and Testing Data\n"
training_size = 200
eigens = 100
#load the data
X_train,X_test, y_train, y_test = load_data()

print "Finding the eigendigits and eigenvectors\n"
# find the eigendigits eigvec(784 x 600) and the mean vector M (784 x 1)
# from chosen t = 0.01 = ratio of chosen data to total training data
t = float(training_size)/60000
A,A_t = chooseTrainingData(X_train, y_train, t) #dimension(A) = 784 x 600
eigvec, mean_vec = findEigenDigits(A)

print "Plotting the original digits and  eigendigits\n"
# plot original image and eigendigit image
n_col = 4
n_row = 4
eigImages = eigvec.T

plot_image('Original Image', A_t[:n_col * n_row], n_col, n_row)
plot_image('Eigendigits', eigImages[:n_col * n_row], n_col, n_row)


print "Reconstructing the original image from the eigenspace projection\n"
# get the reconstruction of the image
n = eigens# Number of eigenvectors considered for reconstruction
rec_num = 16 # Number of images for reconstruction
image_sample = X_test[0:rec_num,:] # rec_num x 784 image samples
image_reconstructed = reconstruction(image_sample, eigvec, mean_vec, n)


print "Plotting the original image and reconstructed image\n"
# plot the original image and the reconstructed image
plot_image('Image before Reconstruction', image_sample[:rec_num], n_col, n_row)
plot_image('Reconstructed Images', image_reconstructed.T[:n_col * n_row], n_col, n_row)

print "Projecting all training and test data to the eigenspace\n"
#project the data to the space of eigenvectors
X_train_projected,X_test_projected = project_data(X_train, X_test, eigvec, mean_vec)

print "Fetching the training sample from training data to train upon\n"
## Get the training set data for training the model.
sample_size = 40 * float(training_size)/60000
_, X_train_sample,_, y_train_sample = \
        train_test_split(X_train_projected, y_train, test_size=sample_size, random_state=49)

print "Fetching the easy and hard test data\n"
###### Get the hard test and easy test data
X_test_hard = X_test_projected[0:5000,:]
y_test_hard = y_test[0:5000,:]
X_test_easy = X_test_projected[5000:10000,:]
y_test_easy = y_test[5000:10000,:]


#### KNN ####
print "Training the K- Nearest neighbor model\n"
neighbors = 5
prediction_easy_KNN,prediction_hard_KNN,accuracy_KNN_easy, accuracy_KNN_hard = \
        KNeighborsClassification(X_train_sample, y_train_sample, X_test_easy, \
        y_test_easy, X_test_hard, y_test_hard, neighbors)


#### Logistic Regression ####
print "Training the Logistic Regression Model\n"
prediction_easy_Logic,prediction_hard_Logic, accuracy_Logistic_easy, accuracy_Logistic_hard = \
        LogisticClassification(X_train_sample,y_train_sample, X_test_easy, \
        y_test_easy, X_test_hard, y_test_hard)


#### Random Forest Classification ####
print "Training the Random Forest model\n"
prediction_easy_Forest,prediction_hard_Forest, accuracy_Forest_easy, accuracy_Forest_hard = \
        ForestClassification(X_train_sample,y_train_sample, X_test_easy, \
        y_test_easy, X_test_hard, y_test_hard)

#### SVM with gaussian kernel ####
print "Training the Support Vector Machine\n"
prediction_easy_SVM,prediction_hard_SVM, accuracy_SVM_easy, accuracy_SVM_hard= \
        SvmClassification(X_train_sample,y_train_sample, X_test_easy, \
        y_test_easy, X_test_hard, y_test_hard)

#END#

# show all the plots
### Uncomment the plots and add a for loop around the #START# and #END# symbol in the code
### With proper indentation to check for all possible values of training data
'''
plt.figure()
plt.plot(range(0,40,4),accuracy_KNN[0,0::], label="KNN")
plt.plot(range(0,40,4),accuracy_SVM[0,0::], label="SVC-RBF")
plt.plot(range(0,40,4),accuracy_Logic[0,0::], label="Logistic Regression")
plt.plot(range(0,40,4),accuracy_Forest[0,0::], label="Random forest")
plt.legend(loc='lower right')
plt.title("Easy Data (k,n_eig,t_eig)=(5,100,200)")
plt.ylabel("Test Set Accuracy")
plt.xlabel("Fit Factor(fit_factor)")

plt.figure()
plt.plot(range(0,40,4),accuracy_KNN[1,0::], label="KNN")
plt.plot(range(0,40,4),accuracy_SVM[1,0::], label="SVC-RBF")
plt.plot(range(0,40,4),accuracy_Logic[1,0::], label="Logistic Regression")
plt.plot(range(0,40,4),accuracy_Forest[1,0::], label="Random forest")
plt.legend(loc= 'lower right')
plt.title("Hard Data (k,n_eig,t_eig)=(5,100,200)")
plt.ylabel("Test Set Accuracy")
plt.xlabel("Fit Factor(fit_factor)")
'''
plt.show()
