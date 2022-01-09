import pandas as pd
import os
from booknlp.booknlp import BookNLP

def get_supersenses(filename):
    model_params={
        "pipeline": "entity,supersense,event", 
        "model": "small"
    }

    booknlp = BookNLP('en', model_params)

    # File identifier
    file_id = '.'.join(filename.split('.')[:-1])

    # Input file to process
    input_file=f'./../../PurePlainDataset/output/{filename}'

    # Output directory to store resulting files in
    output_directory=f'output/{file_id}/'

    # Publish results
    booknlp.process(input_file, output_directory, file_id)

    # Gather results
    df = pd.read_csv(f'{output_directory}{file_id}.tokens', sep='\t')

    # Gather supersenses
    supersenses = pd.read_csv(f'{output_directory}{file_id}.supersense', sep='\t')

    # Begin merging both dataframes
    df['supersense_category'] = ''

    # Loop over supersense df
    for index, row in supersenses.iterrows():
        # Add supersense to all rows in range of starting and ending tokens
        for token in range(row.start_token, row.end_token + 1):
            df.loc[df.token_ID_within_document == token, 'supersense_category'] = row.supersense_category
    
    # Gather entities
    supersenses = pd.read_csv(f'{output_directory}{file_id}.entities', sep='\t')

    # Begin merging entities
    df['prop'] = ''
    df['cat'] = ''
    df['COREF'] = np.nan

    # Loop over entity df
    for index, row in supersenses.iterrows():
        # Add entities to all rows in range of starting and ending tokens
        for token in range(row.start_token, row.end_token + 1):
            df.loc[df.token_ID_within_document == token, ['prop', 'cat', 'COREF']] = [row.prop, row['cat'], row.COREF]
    
    sent_groups = df.groupby('sentence_ID')
    return [sent_groups.get_group(x) for x in sent_groups.groups]


def is_important_for_class(supersense_df):
    # Rule 1: there needs to be at least two nouns in the sentences
    if supersense_df['fine_POS_tag'].tolist().count('NN') + supersense_df['fine_POS_tag'].tolist().count('NNP') + supersense_df['fine_POS_tag'].tolist().count('NNS') > 1:
        if all(item in supersense_df['dependency_relation'].tolist() for item in ['aux', 'auxpass', 'conj']) and 'verb.cognition' in supersense_df['supersense_category'].tolist():
            return True
        if all(item in supersense_df['dependency_relation'].tolist() for item in ['pobj', 'prep', 'nsubj']) and all(item in supersense_df['supersense_category'].tolist() for item in ['verb.stative', 'noun.relation']):
            return True
        if all(item in supersense_df['dependency_relation'].tolist() for item in ['pobj', 'nummod']) and all(item in supersense_df['supersense_category'].tolist() for item in ['verb.stative', 'noun.artifact']) and supersense_df['supersense_category'].tolist().count('verb.stative') > 1:
            return True
        if all(item in supersense_df['dependency_relation'].tolist() for item in ['nsubj', 'dobj', 'amod']) and 'verb.stative' in supersense_df['supersense_category'].tolist() and 'JJ' in supersense_df['fine_POS_tag'].tolist():
            return True
        if all(item in supersense_df['dependency_relation'].tolist() for item in ['det', 'nsubj', 'dobj']) and 'verb.possession' in supersense_df['supersense_category'].tolist():
            return True
        if all(item in supersense_df['dependency_relation'].tolist() for item in ['det', 'nsubj', 'pobj', 'aux']) and 'verb.stative' in supersense_df['supersense_category'].tolist():
            return True
        if all(item in supersense_df['dependency_relation'].tolist() for item in ['det', 'nsubj', 'pobj', 'aux']) and 'verb.communication' in supersense_df['supersense_category'].tolist():
            return True
        if all(item in supersense_df['dependency_relation'].tolist() for item in ['det', 'nsubj', 'dobj', 'cc']) and 'verb.change' in supersense_df['supersense_category'].tolist():
            return True
        if all(item in supersense_df['dependency_relation'].tolist() for item in ['det', 'nsubj', 'dobj']) and all(item in supersense_df['supersense_category'].tolist() for item in ['verb.perception', 'noun.artifact']):
            return True
        if all(item in supersense_df['dependency_relation'].tolist() for item in ['prep', 'nsubjpass', 'auxpass', 'nummod', 'cc']):
            return True
        if all(item in supersense_df['dependency_relation'].tolist() for item in ['nsubjpass', 'auxpass']) and 'require' in supersense_df['lemma'].tolist():
            return True
        if all(item in supersense_df['dependency_relation'].tolist() for item in ['nsubjpass', 'auxpass', 'det', 'aux']) and 'verb.change' in supersense_df['supersense_category'].tolist():
            return True
        if all(item in supersense_df['dependency_relation'].tolist() for item in ['det', 'nsubjpass', 'aux', 'auxpass', 'pobj']) and 'verb.contact' in supersense_df['supersense_category'].tolist():
            return True
        if all(item in supersense_df['dependency_relation'].tolist() for item in ['det', 'nsubj', 'pobj', 'prep']) and 'verb.stative' in supersense_df['supersense_category'].tolist():
            return True
        if all(item in supersense_df['dependency_relation'].tolist() for item in ['pobj', 'prep']) and all(item in supersense_df['supersense_category'].tolist() for item in ['verb.stative', 'noun.artifact']) and supersense_df['dependency_relation'].tolist().count('pobj') > 1:
            return True
        if all(item in supersense_df['dependency_relation'].tolist() for item in ['det', 'nsubj', 'dobj', 'predet']) and 'verb.social' in supersense_df['supersense_category'].tolist():
            return True
        else:
            return False
    else:
        return False

def apply_bucketing(file):
    sentences_supersenses = get_supersenses(file)
    
    bucketed_data = {
        'classes': list(filter(lambda x: is_important_for_class(x), sentences_supersenses))
    }
    
    return {
        bucket: [' '.join(df['token'].values) for df in bucketed_data[bucket]]
        for bucket in bucketed_data.keys()
    }