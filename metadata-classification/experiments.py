'''
Python script for running metadata classification on ALICE computing facility
'''

import socket
from time import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from hypopt import GridSearch
from sklearn import metrics

# Adapted from https://github.com/MeMartijn/FakeNewsDetection
from data_loader import DataLoader  

if __name__ == '__main__':
    # Get starting time of script
    start_time = time()

    print('Python test started on {}'.format(socket.gethostname()))

    # Initialize data loading module
    data = DataLoader()

    # Get word embeddings
    print('Start loading BERT...')
    bert = data.get_bert()

    # Apply pooling strategy
    print('Apply max pooling to dataset...')
    pooled_df = data.apply_pooling('max', bert[['embedding', 'type']])

    # Split training data
    X_train, X_test, y_train, y_test = train_test_split(pooled_df['embedding'].values, pooled_df['type'].values, test_size=0.25, random_state=0)
    
    # Start classification training
    print('Begin training...')
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
    gs = GridSearch(model = LogisticRegression(penalty = 'l2'), param_grid = param_grid)
    gs.fit(X_train, y_train)

    print('Get accuracy over test set...')
    score = gs.score(X_test, y_test)

    print('Ending accuracy over test set:', score)

    print('Generating classification report...')
    print(metrics.classification_report(X_test, y_test))

    print('Python test finished (running time: {0:.1f}s)'.format(time() - start_time))