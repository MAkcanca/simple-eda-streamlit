import itertools
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV

class BaseModelGridSearch:
    def __init__(self, X_train, Y_train, scorer):
        self.X_train = X_train
        self.Y_train = Y_train
        self.scorer = scorer

    def calculate_total_combinations(self, param_grid):
        all_names = sorted(param_grid)
        combinations = itertools.product(*(param_grid[name] for name in all_names))
        return sum(1 for _ in combinations)

    def grid_search(self, model, param_grid):
        current_progress = 0

        def custom_scorer(estimator, X, y_true):
            nonlocal current_progress
            score = self.scorer(estimator, X, y_true)
            current_progress += 1
            return score

        grid_search = GridSearchCV(model, param_grid, cv=5, scoring=custom_scorer)
        grid_search.fit(self.X_train, self.Y_train)
        return grid_search.best_estimator_


class KNNGridSearch(BaseModelGridSearch):
    def search(self, param_grid):
        return self.grid_search(KNeighborsClassifier(), param_grid)

class SVMGridSearch(BaseModelGridSearch):
    def search(self, param_grid):
        return self.grid_search(SVC(), param_grid)

class NBGridSearch(BaseModelGridSearch):
    def search(self, param_grid):
        return self.grid_search(GaussianNB(), param_grid)