import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn import svm
from sklearn.metrics import confusion_matrix
from scipy.stats import uniform
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt



def cross_validate(trainData, trainLabels, kernel):
    model = svm.SVC()
    C = np.logspace(-2, 10, 13)
    param_distribution = {'kernel': [kernel], 'C': C}
    num_iter = len(C)
    if (kernel == 'rbf'):
        param_distribution['gamma'] = np.logspace(-9, 3, 26)
        num_iter = 100
    print('starting cv')
    selection = RandomizedSearchCV(model, param_distribution, n_iter = num_iter, n_jobs=-1, verbose=1, cv=3)
    selection.fit(trainData, trainLabels)
    return selection.best_params_


# Pretty-prints confusion matrix (taken off S.O.)
def confusion(labels, prediction, classes):
    cm = confusion_matrix(labels, prediction)
    df_cm = pd.DataFrame(cm, classes, classes)
    # plt.figure(figsize=(10,7))
    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})  # font size
    plt.show()