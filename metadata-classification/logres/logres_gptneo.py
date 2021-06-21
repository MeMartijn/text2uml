'''
Python script for running metadata classification on ALICE computing facility
'''

import socket
from time import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from hypopt import GridSearch
from sklearn.metrics import classification_report
import pandas as pd
import sys

# Adapted from https://github.com/MeMartijn/FakeNewsDetection
sys.path.append('../')
from data_loader import DataLoader  

if __name__ == '__main__':
    # Get starting time of script
    start_time = time()

    print('Python test started on {}'.format(socket.gethostname()))

    # Initialize data loading module
    data = DataLoader()

    # Get word embeddings
    print('Start loading GPT NEO...')
    gpt = data.get_gptneo()

    # Apply pooling strategy
    for pooling_strategy in ['max', 'average', 'min']:
        print(f'Running experiment with {pooling_strategy} pooling')

        print(f'Apply {pooling_strategy} pooling to dataset...')
        pooled_df = data.apply_pooling('max', gpt[['embedding', 'type']])

        # Split training data
        print('Generate train and test sets')
        X_train, X_test, y_train, y_test = train_test_split(pooled_df['embedding'].to_list(), pooled_df['type'].to_list(), test_size=0.25, random_state=0)

        # Delete dataset to save memory
        del pooled_df
        
        # Start classification training
        print('Begin training...')
        param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
        gs = GridSearch(model = LogisticRegression(penalty = 'l2'), param_grid = param_grid)
        gs.fit(X_train, y_train)

        print('Predict test set...')
        y_pred = gs.predict(X_test)

        print('Generating classification report...')
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        pd.DataFrame(report_dict).to_csv(f'{pooling_strategy}_gptneo_logres.csv', index=False)

        # Delete big items from memory
        del X_train
        del X_test
        del y_train
        del y_test
        del gs

    print('Python test finished (running time: {0:.1f}s)'.format(time() - start_time))