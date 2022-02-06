from booknlp.booknlp import BookNLP
import os
import pandas as pd
import csv
import numpy as np

class Text2UML:
    def __init__(self, source):
        # Set source document
        self.source = source
        self.id = '-'.join(os.path.basename(source).split('.')[:-1])
        self.output_dir = f'output/{self.id}'

    def generate_text_data(self):
        # Initialize BookNLP
        booknlp = BookNLP('en', {
            'pipeline': 'entity,quote,supersense,event,coref',
            'model': 'big', 
        })

        # Process source text
        booknlp.process(self.source, self.output_dir, self.id)

        # Put process data into memory
        self.tokens = pd.read_csv(f'{self.output_dir}/{self.id}.tokens', sep='\t', quoting=csv.QUOTE_NONE)
        self.get_supersenses()
        self.detected_entities = self.get_detected_entities()
    
    def get_supersenses(self):
        # Get supersense table
        supersenses = pd.read_csv(f'{self.output_dir}/{self.id}.supersense', sep='\t')

        # Merge supersenses with main token table
        self.tokens['supersense_category'] = ''

        # Loop over supersense df
        for index, row in supersenses.iterrows():
            # Add supersense to all rows in range of starting and ending tokens
            for token in range(row.start_token, row.end_token + 1):
                self.tokens.loc[self.tokens.token_ID_within_document == token, 'supersense_category'] = row['supersense_category']

    def get_detected_entities(self):
        # Get entities table
        entities = pd.read_csv(f'{self.output_dir}/{self.id}.entities', sep='\t')

        # Merge entities with main token table
        self.tokens['entity'] = np.nan
        self.tokens['entity_type'] = ''
        self.tokens['entity_category'] = ''

        # Loop over entity df
        for index, row in entities.iterrows():
            # Add entity data to all rows in range of starting and ending tokens
            for token in range(row.start_token, row.end_token + 1):
                self.tokens.loc[self.tokens.token_ID_within_document == token, 'entity'] = row.COREF
                self.tokens.loc[self.tokens.token_ID_within_document == token, 'entity_type'] = row.prop
                self.tokens.loc[self.tokens.token_ID_within_document == token, 'entity_category'] = row['cat']

        # Only return the entities as a dictionary
        return {
            row['COREF']: row['text']
            for index, row in entities.drop_duplicates('COREF').iterrows()
        }

    def get_main_entity(self, df):
        # Single-word entities
        if len(df) == 1:
            # If it's only a one-off entity, keep it
            return df
        # Multi-word entities
        else:
            # Check if token positions are sequential (and therefore one entity over multiple words)
            token_positions = df['token_ID_within_document'].to_list()

            if list(range(token_positions[0], token_positions[-1] + 1)) == token_positions:
                # One entity over multiple words
                return df
            else: 
                # One entity in multiple positions
                # First: drop duplicates
                filtered_entities = df.drop_duplicates('lemma')

                # Get all entities
                grouped_entities = []
                last_token_id = np.nan
                current_group = 0
                i = 0

                while last_token_id != filtered_entities.iloc[-1]['token_ID_within_document']:
                    current_row = filtered_entities.iloc[i]
                    current_token_id = current_row['token_ID_within_document']

                    if not np.isnan(last_token_id):
                        if current_token_id == last_token_id + 1:
                            # It belongs to the previous group
                            grouped_entities[current_group] = grouped_entities[current_group].append(current_row, ignore_index=True)
                        else:
                            # It belongs to a new group
                            current_group += 1
                            grouped_entities.append(current_row.to_frame().T)
                    else:
                        # First element
                        grouped_entities.append(current_row.to_frame().T)

                    # Go to next row
                    i += 1
                    last_token_id = current_token_id

                # Check which is the "main" entity
                main_entity = list(filter(lambda x: ' '.join(x['word']) in self.detected_entities.values(), grouped_entities))[0]

                return main_entity
    
    def get_filtered_entities(self):
        # Define mask for specific types of entities that shouldn't be filtered
        type_mask = ((self.tokens['entity_type'] != 'PRON') & (self.tokens['entity_category'] != 'ORG') & (self.tokens['entity_category'] != 'GPE'))

        # Remove organisations, these remain as full entity
        filtered_entities = list(self.tokens[(~self.tokens['entity'].isnull()) & (type_mask)].groupby('entity'))

        # Util function to get the sentence out of a dataframe
        get_sentence = lambda df: ' '.join(df['lemma'])

        # Gather final list of transformed entities and their related entity id
        final_entities = {}
        for entity in filtered_entities:
            main_entity = self.get_main_entity(entity[1])

            # Only keep important POS tags
            if len(main_entity) > 1:
                main_entity = main_entity[(main_entity['POS_tag'] == 'NOUN') | (main_entity['POS_tag'] == 'ADJ') | (main_entity['POS_tag'] == 'PRP')]

            # Add to final dictionary
            final_entities.setdefault(get_sentence(main_entity), []).append(entity[0])
        
        # Add non-used entities from the main entity list
        located_entities = [item for sublist in final_entities.values() for item in sublist]
        for entity_id, entity_name in self.detected_entities.items():
            if entity_id not in located_entities:
                final_entities.setdefault(entity_name.lower(), []).append(float(entity_id))
        
        return final_entities
    
    def get_filtered_text(self, requested_entities):
        # Get all the correct entity numbers
        all_entities = self.get_filtered_entities()
        selected_entities = [item for sublist in [all_entities[entity] for entity in requested_entities] for item in sublist]

        filtered_sentences = list(filter(lambda df: any(df[1]['entity'].isin(selected_entities)), self.tokens.groupby('sentence_ID')))
        return list(map(lambda sentence: ' '.join(sentence[1]['word']), filtered_sentences))

request = Text2UML('rental-truck.txt')
request.generate_text_data()