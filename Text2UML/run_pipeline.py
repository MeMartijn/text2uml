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
    def __init__(self, source_file = None, input_text = None):
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
        self.summarization = Summarizer(self.tokens, self.entities, self.set_saved_text)
        self.bucketing = Bucketizer(self.tokens, self.get_saved_text, self.set_saved_text)
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
        tokens = pd.read_csv(f'{self.output_dir}/{self.id}.tokens', sep='\t', quoting=csv.QUOTE_NONE)
        tokens = self.get_supersenses(tokens)
        tokens = self.get_detected_entities(tokens)

        return tokens
    
    def get_supersenses(self, tokens):
        # Get supersense table
        supersenses = pd.read_csv(f'{self.output_dir}/{self.id}.supersense', sep='\t')

        # Merge supersenses with main token table
        tokens['supersense_category'] = ''

        # Loop over supersense df
        for index, row in supersenses.iterrows():
            # Add supersense to all rows in range of starting and ending tokens
            for token in range(row.start_token, row.end_token + 1):
                tokens.loc[tokens.token_ID_within_document == token, 'supersense_category'] = row['supersense_category']

        return tokens

    def get_detected_entities(self, tokens):
        # Get entities table
        entities = pd.read_csv(f'{self.output_dir}/{self.id}.entities', sep='\t')

        # Merge entities with main token table
        tokens['entity'] = np.nan
        tokens['entity_type'] = ''
        tokens['entity_category'] = ''

        # Loop over entity df
        for index, row in entities.iterrows():
            # Add entity data to all rows in range of starting and ending tokens
            for token in range(row.start_token, row.end_token + 1):
                tokens.loc[tokens.token_ID_within_document == token, 'entity'] = row.COREF
                tokens.loc[tokens.token_ID_within_document == token, 'entity_type'] = row.prop
                tokens.loc[tokens.token_ID_within_document == token, 'entity_category'] = row['cat']

        # Only return the entities as a dictionary
        self.entities = {
            row['COREF']: row['text']
            for index, row in entities.drop_duplicates('COREF').iterrows()
        }

        return tokens

request = Text2UML(input_text = '''
    Romano's is the finest Italian restaurant in the city.
Unless you are a celebrity or a good friend of Romano you will need a reservation.
A reservation is made for a specific time, date and number of people.
The reservation also captures the name and phone number of the person making the reservation.
Each reservation is assigned a unique reservation number.
There are two categories of reservations at Romano's: individual reservations and banquet reservations.
Additional reservation information captured when an individual makes a reservation includes seating preference (inside or patio) and smoking preference (smoking or nonsmoking).
Additional reservation information captured for banquet reservations includes the group name and the method of payment.
Seating at Romano's is limited.
Romano's has a fixed number of tables.
Each table is identified by a unique table number.
Each of the tables is further described by a unique free form description such as "located by the North window", "located in front of the fountain", "by the kitchen door".
Each table is classified as a 2-person, 4-person or 6-person table.
When a reservation is made, Romano associates a specific number to the reservation.
A table can be utilized many times over the evening by many reservations.
Romano tends to overbook tables.
Therefore, there can be overlapping table reservations.
The management structure at Romano's is hierarchical.
There are several restaurant managers who report to Romano.
The managers are responsible for managing the Maitre'd and the chefs as well as ensuring that the guests have a pleasant dining experience.
The Maitre'd is responsible for managing the waiters, bartenders and bus personnel.
The Chefs are responsible for managing the cooks and dishwashers.
Each person working for Romano's must be classified as either a manager, Maitre'd, waiter, bartender, chef, cook, bus person or dishwasher.
Additional information maintained by Romano's for each person includes the persons name, date of birth and drivers license number.
When the reservation party arrives at Romano's the reservation is assigned to one waiter.
A waiter can be assigned to many reservations during the course of the evening".
The menu at Romano's is exquisite.
There are many exciting and exotic items.
Each menu item is identified by a unique menu item number.
Information maintained by Romano's for each menu item includes an item description of (e.g. "chicken marsala", "fish soup", "endive salad","1988 Merlot wine", etc.), and item prep time.
Each menu item is classified by Romano's as "appetizer", "entree", "dessert" or "beverage".
The price of each menu item can vary based on the time of day.
For example, some of the menu items have different lunch and dinner prices.
Some of the menu items change prices for happy hour.
In order to calculate the check at the end of the dinner, the waiter maintains a list, by reservation number, of the menu items ordered and the time that the menu item was ordered.
In other words, each reservation can be associated with many menu items and a menu item can be associated with many reservations.
In addition to menu items, Romano's maintains a list of the food items that are utilized by the restaurant such as chicken, mushrooms, bread sticks, red sauce, cream sauce, etc.
Food items are utilized in the preparation of menu items.
Each food item is identified by a unique food item number.
''')
request.summarization.get_filtered_entities()
request.summarization.get_filtered_text(['customer', 'vehicle', 'truck'])
request.bucketing.apply_bucketing()
request.classification.get_prediction()