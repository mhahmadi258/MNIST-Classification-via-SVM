import numpy as np
import matplotlib.pyplot as plt
import gzip
from sklearn.metrics import ConfusionMatrixDisplay, precision_score, recall_score, f1_score, roc_curve, auc


def decode_mnist(images_path, labels_path):
    """
    Extract MNIST images and their corresponding labels by getting path of images file and labels files.    
    """
    with gzip.open(labels_path, 'rb') as f:
        labels = np.frombuffer(f.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as f:
        images = np.frombuffer(f.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 28*28)

    return images, labels


def image_show(img):
    I = img.reshape(28, 28)
    plt.imshow(I, cmap='gray')
    plt.show()


def evaluate(classifire, x, y_true, roc_method):
    """
    Evaluate and display the results of evaluations. The model is evaluated by multiple metric such as precisions and recall and AUC.
    """
    ConfusionMatrixDisplay.from_estimator(classifire, x, y_true)
    ys = classifire.label_binarizer_.transform(y_true).toarray().T
    figure, axis = plt.subplots(4, 3, figsize=(9, 9))
    precisions = list()
    recalls = list()
    f1s = list()
    for i, (classifier, y) in enumerate(zip(classifire.estimators_, ys)):
        y_pred = classifier.predict(x)
        precision = precision_score(y, y_pred)
        precisions.append(precision)
        recall = recall_score(y, y_pred)
        recalls.append(recall)
        f1 = f1_score(y, y_pred)
        f1s.append(f1)
        print(
            f'class {i} : precision = {precision},\t recall = {recall},\t f1 = {f1}')
        if i == 9:
            print(
                f'total : precision = {np.mean(precisions)},\t recall = {np.mean(recalls)},\t f1 = {np.mean(f1s)}\n')
        if roc_method == 'decision':
            y_score = classifier.decision_function(x)
        else:
            y_score = classifier.predict(x)
        fp, tp, th = roc_curve(y, y_score)
        axis[i//3, i % 3].plot(fp, tp)
        axis[i//3, i % 3].plot([0, 1], [0, 1], 'k--')
        axis[i//3, i % 3].set_title(f'class {i} : {auc(fp,tp)}')
        axis[i//3, i % 3].set_xlabel('False Positive Rate')
        axis[i//3, i % 3].set_ylabel('True Positive Rate')
    figure.tight_layout()
    plt.show()
