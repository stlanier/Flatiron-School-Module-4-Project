import numpy as np
import sklearn
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, plot_roc_curve
import keras.backend as K
import matplotlib.pyplot as plt

def report(y, yhat):
    cm = np.round(confusion_matrix(y, yhat, normalize='true'), 3)*100
    print(cm)
    print()
    print(classification_report(y, yhat))


# I'm leaning strongly towards using F1 scores, but Keras doesn't include that in its metrics. I got the following code from [here](https://medium.com/@aakashgoel12/how-to-add-user-defined-function-get-f1-score-in-keras-metrics-3013f979ce0d) so that Keras can include the F1 score as a metric.

def get_f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


def roc_curve_multiclass(clf, X, y_bin, axis=None, title=None):
    y_score = 0
    if ((type(clf) == sklearn.linear_model._logistic.LogisticRegressionCV) | 
        (type(clf) == sklearn.svm._classes.SVC)):
        y_score = clf.decision_function(X)
    else:
        y_score = clf.predict_proba(X)
    fpr = {}
    tpr = {}
    roc_auc = dict()
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    if axis:
        for i in range(3):
            axis.plot(fpr[i], tpr[i], label=str(i))
        axis.legend()
        axis.set_xlabel('FPR')
        axis.set_ylabel('TPR')
        axis.set_title(title)
        
    else:
        plt.plot(figsize=(10,10))
        for i in range(3): 
            plt.plot(fpr[i], tpr[i], label=str(i))
        plt.legend()
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title(title)
