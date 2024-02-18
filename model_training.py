import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics._scorer import accuracy_scorer
from config import NB_PARAM_GRID, SVM_PARAM_GRID, KNN_PARAM_GRID

class ModelTrainer:
    def __init__(self, model_type, preprocessor=None):
        self.model_type = model_type
        self.preprocessor = preprocessor
        self.best_model = None
        self.progress_bar = None
        self.total_progress = 0
        self.current_progress = 0
        self.param_grid = None
        self.set_param_grid()

    def set_param_grid(self):
        dict = {
            "KNN": KNN_PARAM_GRID,
            "SVM": SVM_PARAM_GRID,
            "Naive Bayes": NB_PARAM_GRID
        }
        if self.model_type not in dict:
            raise ValueError("Invalid model type")
        self.param_grid = dict[self.model_type]

    def set_progress_bar(self, progress_bar):
        self.progress_bar = progress_bar

    def my_accuracy_scorer(self, estimator, X, y_true):
        score = accuracy_scorer(estimator, X, y_true)
        self.current_progress += 1
        if self.progress_bar:
            self.progress_bar.progress(self.current_progress / self.total_progress)
        return score

    def calculate_total_combinations(self):
        all_names = sorted(self.param_grid)
        combinations = itertools.product(*(self.param_grid[name] for name in all_names))
        self.total_progress = sum(1 for _ in combinations) * 5

    def train_model(self, X_train, y_train, multiprocess=False):
        if self.model_type == "KNN":
            estimator = KNeighborsClassifier()
        elif self.model_type == "SVM":
            estimator = SVC(probability=True)
        else:  # Naive Bayes
            estimator = GaussianNB()

        self.calculate_total_combinations()
        self.current_progress = 0
        if multiprocess:
            grid_search = GridSearchCV(estimator, self.param_grid, cv=5, verbose=3, n_jobs=-1)
        else:
            grid_search = GridSearchCV(estimator, self.param_grid, cv=5, verbose=3, scoring=self.my_accuracy_scorer)
        grid_search.fit(X_train, y_train)
        self.best_model = grid_search.best_estimator_

    def evaluate_model(self, X_test, y_test):
        y_pred = self.best_model.predict(X_test)
        y_probs = self.best_model.predict_proba(X_test)[:, 1] if hasattr(self.best_model, "predict_proba") else None
        average = 'binary' if len(np.unique(y_test)) <= 2 else 'macro'
        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average=average),
            "Recall": recall_score(y_test, y_pred, average=average),
            "F1 Score": f1_score(y_test, y_pred, average=average)
        }
        cm = confusion_matrix(y_test, y_pred)
        return metrics, cm, y_probs

