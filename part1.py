import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


def cross_validate(trainData, trainLabels, kernel, C=0):
    model = svm.SVC()
    C = np.logspace(-10, 10, num=21) if kernel == 'linear' else C
    params = {'kernel': [kernel], 'C': C}
    if (kernel == 'RBF'):
        params['gamma'] = np.logspace(-10, 10, num=21)
    selection = GridSearchCV(model, params, n_jobs=-1, verbose=1)
    selection.fit(trainData, trainLabels)
    return selection.best_params_

# Pretty-prints confusion matrix (taken off s.o.)
def confusion(labels, prediction, classes):
    cm = confusion_matrix(labels, prediction)
    df_cm = pd.DataFrame(cm, classes, classes)
    # plt.figure(figsize=(10,7))
    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})  # font size
    plt.show()