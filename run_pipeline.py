from booknlp.booknlp import BookNLP
import os
import pandas as pd
import csv
import numpy as np
from nltk.tokenize import sent_tokenize
import uuid

# Module imports
from summarization import Summarizer
from bucketing import Bucketizer
from classification import Classifier


class Text2UML:
    def __init__(self, source_file=None, input_text=None):
        if source_file:
            self.source = source_file
        else:
            # This system works on plaintext files, create a plaintext file from input text
            sentences = sent_tokenize(input_text)

            # Create input folder if it doesn't exist
            if not os.path.exists('input'):
                os.makedirs('input')

            # Write all sentences to txt file
            input_file = f'input/{uuid.uuid4()}.txt'
            with open(input_file, 'w') as f:
                for line in sentences:
                    f.write('%s\n' % line)

            self.source = input_file

        # Set source document
        self.id = '-'.join(os.path.basename(self.source).split('.')[:-1])
        self.output_dir = f'output/{self.id}'

        # Generate text data
        self.tokens = self.generate_text_data()

        # Attach modules
        self.summarization = Summarizer(
            self.tokens, self.entities, self.set_saved_text)
        self.bucketing = Bucketizer(
            self.tokens, self.get_saved_text, self.set_saved_text)
        self.classification = Classifier(self.tokens, self.get_saved_text)

    def set_saved_text(self, text):
        self.saved_text = text

    def get_saved_text(self):
        return self.saved_text

    def generate_text_data(self):
        # Initialize BookNLP
        booknlp = BookNLP('en', {
            'pipeline': 'entity,quote,supersense,event,coref',
            'model': 'big',
        })

        # Process source text
        booknlp.process(self.source, self.output_dir, self.id)

        # Aggregate all process data tables into one
        tokens = pd.read_csv(
            f'{self.output_dir}/{self.id}.tokens', sep='\t', quoting=csv.QUOTE_NONE)
        tokens = self.get_supersenses(tokens)
        tokens = self.get_detected_entities(tokens)

        return tokens

    def get_supersenses(self, tokens):
        # Get supersense table
        supersenses = pd.read_csv(
            f'{self.output_dir}/{self.id}.supersense', sep='\t')

        # Merge supersenses with main token table
        tokens['supersense_category'] = ''

        # Loop over supersense df
        for index, row in supersenses.iterrows():
            # Add supersense to all rows in range of starting and ending tokens
            for token in range(row.start_token, row.end_token + 1):
                tokens.loc[tokens.token_ID_within_document == token,
                           'supersense_category'] = row['supersense_category']

        return tokens

    def get_detected_entities(self, tokens):
        # Get entities table
        entities = pd.read_csv(
            f'{self.output_dir}/{self.id}.entities', sep='\t')

        # Merge entities with main token table
        tokens['entity'] = np.nan
        tokens['entity_type'] = ''
        tokens['entity_category'] = ''

        # Loop over entity df
        for index, row in entities.iterrows():
            # Add entity data to all rows in range of starting and ending tokens
            for token in range(row.start_token, row.end_token + 1):
                tokens.loc[tokens.token_ID_within_document ==
                           token, 'entity'] = row.COREF
                tokens.loc[tokens.token_ID_within_document ==
                           token, 'entity_type'] = row.prop
                tokens.loc[tokens.token_ID_within_document ==
                           token, 'entity_category'] = row['cat']

        # Only return the entities as a dictionary
        self.entities = {
            row['COREF']: row['text']
            for index, row in entities.drop_duplicates('COREF').iterrows()
        }

        return tokens
