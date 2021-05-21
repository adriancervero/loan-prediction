from os import X_OK
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

import config as cfg

models = {
    'logreg': LogisticRegression(max_iter=1000,class_weight="balanced"),
    'rf': RandomForestClassifier(max_depth=6, random_state=cfg.RANDOM_STATE),
    'svm': SVC(),
    'knn': KNeighborsClassifier(),
}

param_grid = {
    'logreg':{
        "model__C":[0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
        "model__solver": ['newton-cg', 'lbfgs']
    }
}