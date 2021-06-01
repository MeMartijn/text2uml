from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from hypopt import GridSearch

class Classifiers:
    '''Interface for using pre-defined classifiers'''
    @staticmethod
    def get_svm_score(X_train, X_test, X_validation, y_train, y_test, y_validation, penalty = 'l2'):
        param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
        gs = GridSearch(model = LinearSVC(penalty = penalty), param_grid = param_grid)
        gs.fit(X_train, y_train, X_validation, y_validation)

        return gs.score(X_test, y_test)

    @staticmethod
    def get_logres_score(X_train, X_test, X_validation, y_train, y_test, y_validation, penalty = 'l2'):
        param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
        gs = GridSearch(model = LogisticRegression(penalty = penalty), param_grid = param_grid)
        gs.fit(X_train, y_train, X_validation, y_validation)

        return gs.score(X_test, y_test)
    
    @staticmethod
    def get_gradientboosting_score(X_train, X_test, X_validation, y_train, y_test, y_validation):
        param_grid = {'learning_rate': [0.1, 0.05, 0.02, 0.01]}
        gs = GridSearch(model = GradientBoostingClassifier(), param_grid = param_grid)
        gs.fit(X_train, y_train, X_validation, y_validation)

        return gs.score(X_test, y_test)
