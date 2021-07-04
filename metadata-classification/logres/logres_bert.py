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
from classifiers import Classifiers

if __name__ == '__main__':
    # Get starting time of script
    start_time = time()

    print('Python test started on {}'.format(socket.gethostname()))

    # Initialize modules
    data = DataLoader()
    clfs = Classifiers()

    # Get word embeddings
    print('Start loading BERT...')
    bert = data.get_bert()

    # Apply pooling strategy
    for pooling_strategy in ['max', 'average', 'min']:
        print(f'Running experiment with {pooling_strategy} pooling')
        report_dict = clfs.get_logres_scores(data, bert, pooling_strategy, penalty = 'l2')
        pd.DataFrame(report_dict).to_csv(f'{pooling_strategy}_gptneo_logres.csv', index=False)

        print(f'BERT + {pooling_strategy} finished (running time: {0:.1f}s)'.format(time() - start_time))

    print('Python test finished (running time: {0:.1f}s)'.format(time() - start_time))