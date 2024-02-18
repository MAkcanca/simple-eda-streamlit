# Constants for model parameter grids
KNN_PARAM_GRID = {'n_neighbors': [3, 5, 11, 19], 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan']}
SVM_PARAM_GRID = {'C': [0.1, 1, 10, 100], 'kernel': ['rbf', 'poly', 'sigmoid']}
NB_PARAM_GRID = {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]}

