import os
import pandas as pd
import numpy as np
import re
from tqdm.auto import tqdm
import json

# Flair embedding imports
from flair.data import Sentence
from flair.embeddings import TransformerWordEmbeddings
import os
from nltk import tokenize

print('Test on individual sentence')
# Test statements
statement = 'This is a test'
embedding = TransformerWordEmbeddings('bert-base-uncased')

# Split all sentences
sentences = tokenize.sent_tokenize(statement)
print(sentences)

# Create an array for storing the embeddings
vector = []

# Loop over all sentences and apply embedding
for sentence in sentences:
    # Create a Sentence object for each sentence in the statement
    sentence = Sentence(sentence, use_tokenizer = True)

    # Embed words in sentence
    embedding.embed(sentence)
    vector.append([token.embedding.numpy() for token in sentence])

print(vector)

print('Test on full data loader')
data = DataLoader()

print(data.df.iloc[0])

bert = data.get_bert()

print(bert.iloc[0])
