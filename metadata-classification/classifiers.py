from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from hypopt import GridSearch
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from hypopt import GridSearch
from sklearn.metrics import classification_report

class Classifiers:
    '''Interface for using pre-defined classifiers'''
    @staticmethod
    def get_svm_scores(data, df, pooling_strategy, penalty = 'l2'):
        print(f'Apply {pooling_strategy} pooling to dataset...')
        pooled_df = data.apply_pooling(pooling_strategy, df[['embedding', 'type']])

        # Split training data
        print('Generate train and test sets')
        X_train, X_test, y_train, y_test = train_test_split(pooled_df['embedding'].to_list(), pooled_df['type'].to_list(), test_size=0.25, random_state=0)

        # Delete dataset to save memory
        del pooled_df
        
        # Start classification training
        print('Begin training...')
        param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
        gs = GridSearch(model = LinearSVC(penalty = penalty), param_grid = param_grid)
        gs.fit(X_train, y_train)

        print('Predict test set...')
        y_pred = gs.predict(X_test)

        print('Generating classification report...')
        return classification_report(y_test, y_pred, output_dict=True)

    @staticmethod
    def get_logres_scores(data, df, pooling_strategy, penalty = 'l2'):
        print(f'Apply {pooling_strategy} pooling to dataset...')
        pooled_df = data.apply_pooling(pooling_strategy, df[['embedding', 'type']])

        # Split training data
        print('Generate train and test sets')
        X_train, X_test, y_train, y_test = train_test_split(pooled_df['embedding'].to_list(), pooled_df['type'].to_list(), test_size=0.25, random_state=0)

        # Delete dataset to save memory
        del pooled_df
        
        # Start classification training
        print('Begin training...')
        param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
        gs = GridSearch(model = LogisticRegression(penalty = penalty), param_grid = param_grid)
        gs.fit(X_train, y_train)

        print('Predict test set...')
        y_pred = gs.predict(X_test)

        print('Generating classification report...')
        return classification_report(y_test, y_pred, output_dict=True)
    
    @staticmethod
    def get_gradientboosting_scores(data, df, pooling_strategy):
        print(f'Apply {pooling_strategy} pooling to dataset...')
        pooled_df = data.apply_pooling(pooling_strategy, df[['embedding', 'type']])

        # Split training data
        print('Generate train and test sets')
        X_train, X_test, y_train, y_test = train_test_split(pooled_df['embedding'].to_list(), pooled_df['type'].to_list(), test_size=0.25, random_state=0)

        # Delete dataset to save memory
        del pooled_df
        
        # Start classification training
        print('Begin training...')
        param_grid = {'learning_rate': [0.1, 0.05, 0.02, 0.01]}
        gs = GridSearch(model = GradientBoostingClassifier(), param_grid = param_grid)
        gs.fit(X_train, y_train)

        print('Predict test set...')
        y_pred = gs.predict(X_test)

        print('Generating classification report...')
        return classification_report(y_test, y_pred, output_dict=True)
