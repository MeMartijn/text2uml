import numpy as np

class Summarizer:
    def __init__(self, tokens, entities, save_text):
        self.tokens = tokens
        self.entities = entities
        self.save_text = save_text
    
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
                main_entity = list(filter(lambda x: ' '.join(x['word']) in self.entities.values(), grouped_entities))[0]

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
        for entity_id, entity_name in self.entities.items():
            if entity_id not in located_entities:
                final_entities.setdefault(entity_name.lower(), []).append(float(entity_id))
        
        return final_entities
    
    def get_filtered_text(self, requested_entities):
        # Get all the correct entity numbers
        all_entities = self.get_filtered_entities()
        selected_entities = [item for sublist in [all_entities[entity] for entity in requested_entities] for item in sublist]

        filtered_sentences = list(filter(lambda df: any(df[1]['entity'].isin(selected_entities)), self.tokens.groupby('sentence_ID')))

        # Save text for in later processes in the main class
        self.save_text(filtered_sentences)

        return list(map(lambda sentence: ' '.join(sentence[1]['word']), filtered_sentences))
    