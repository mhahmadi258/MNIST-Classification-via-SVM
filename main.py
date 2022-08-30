import numpy as np
import pickle
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score,f1_score
from utils import decode_mnist, image_show, evaluate



if __name__ == "__main__":

    # Reading dataset
    X_train, Y_train = decode_mnist(
        'train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz')
    X_test, Y_test = decode_mnist(
        't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz')

    print(f'train images shape: {X_train.shape}')
    print(f'train labels shape: {Y_train.shape}')
    print(f'test images shape: {X_test.shape}')
    print(f'test labels shape: {Y_test.shape}')

    # Displaying a random sample
    image_index = 7
    print(f'images label : {Y_train[image_index]}')
    image_show(X_train[image_index])

    # Apply feature reduction by PCA
    print('Applying PCA ...')
    pca100 = PCA(n_components=100)
    X_train_pca100 = pca100.fit_transform(X_train)
    X_test_pca100 = pca100.transform(X_test)

    print(f'shape of training set : {X_train_pca100.shape}')
    print(f'shape of test set : {X_test_pca100.shape}')

    # Using svm with linear kernel for classification
    print('Start classification with linear kernel ...')
    clf_linear = OneVsRestClassifier(svm.LinearSVC(C=1.0))
    clf_linear.fit(X_train_pca100, Y_train)
    pickle.dump(clf_linear, open('clf_linear.pkl', 'wb'))
    print('Start evaluation for linear kernel ...')
    linear_score = clf_linear.score(X_test_pca100, Y_test)
    print(f'Score of svm with linear kernel is : {linear_score}')
    evaluate(clf_linear, X_test_pca100, Y_test, 'decision')

    # Using svm with RBF kernel for classification
    print('Start classification with RBF kernel ...')
    clf_rbf = OneVsRestClassifier(svm.SVC(C=1.0, kernel='rbf', gamma='scale'))
    clf_rbf.fit(X_train_pca100, Y_train)
    pickle.dump(clf_linear, open('clf_rbf.pkl', 'wb'))
    print('Start evaluation for RBF kernel ...')
    rbf_score = clf_rbf.score(X_test_pca100, Y_test)
    print(f'Score of svm with RBF kernel is : {linear_score}')
    evaluate(clf_rbf, X_test_pca100, Y_test, 'decision')
    
    # K-fold evaluation
    print('Start training and evaluation with k-fold method ...')
    X = np.vstack((X_train_pca100,X_test_pca100))
    Y = np.hstack((Y_train,Y_test))
    print(f'x shape: {X.shape}')
    print(f'y shape: {Y.shape}')
    kf = KFold(n_splits=5)
    precisions =list()
    recalls = list()
    f1s = list()
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        clf = OneVsRestClassifier(svm.SVC())
        clf.fit(X_train,Y_train)
        y_pred = clf.predict(X_test)
        precisions.append(precision_score(Y_test,y_pred, average='macro'))
        recalls.append(recall_score(Y_test,y_pred, average='macro'))
        f1s.append(f1_score(Y_test,y_pred, average='macro'))
        print(f'precision : {np.mean(precisions)}')
        print(f'recall : {np.mean(recalls)}')
        print(f'f1 : {np.mean(f1s)}')