import spacy
import pandas as pd
import os
from nltk import tokenize

nlp = spacy.load("en_core_web_sm")

def get_supersenses(sent):
    doc = nlp(sent)
    target_file = '../../pysupersensetagger/input'

    if os.path.exists(target_file):
        os.remove(target_file)
        
    with open(target_file, 'a') as inp_file:
        for token in doc:
            inp_file.write(f'{token}\t{token.tag_}\n')
    
    
    os.system(f'cd {"/".join(target_file.split("/")[:-1])} && sh sst.sh {target_file.split("/")[-1]}')
    
    return pd.read_csv('../../pysupersensetagger/input.pred.tags', sep='\t', names=['token', 'lemma', 'POS-tag', 'MWE+supersense tag', 'MWE parent offset', 'MWE attachment strength', 'supersense label', 'sentence ID'])

def get_metadata(supersense_df):
    prepositions = supersense_df[supersense_df['POS-tag'] == 'IN']

    metadata = {
        'objects': []
    }

    # Check if there are groups of nouns
    if len(prepositions) > 0:
        # Define target groups
        targets = ['NN', 'NNS']
            
        for pps in prepositions.iterrows():
            # If ending or beginning position, it can't be a group of nouns
            if not (pps[0] == 0 or pps[0] >= len(supersense_df) - 1):
                # Define surroundings
                preword = supersense_df.loc[pps[0] - 1]
                postword = supersense_df.loc[pps[0] + 1]
                
                # Check if surroundings are in target groups
                if preword['POS-tag'] in targets and postword['POS-tag'] in targets:
                    # Add grouping to metadata
                    metadata['objects'].append(preword.token + ' ' + pps[1].token + ' ' + postword.token)
                    
                    # Drop extracted groups from selection
                    mask = supersense_df.index.isin(list(range(pps[0] -1, pps[0] + 2)))
                    supersense_df = supersense_df[~mask]
            
        # Now get all other nouns and add to metadata
        nouns = supersense_df[supersense_df['POS-tag'].isin(targets)]
        metadata['objects'] = metadata['objects'] + nouns['token'].to_list()
    
    return metadata


def is_important_for_class(supersense_df):
    # Rule 1: there needs to be at least two nouns in the sentences
    if len(supersense_df[supersense_df['POS-tag'] == 'NN']) > 1:
        return True
    else:
        return False

def apply_bucketing(summary):
    sentences = tokenize.sent_tokenize(summary)
    
    sentences_supersenses = [get_supersenses(sentence) for sentence in sentences]
    
    bucketed_data = {
        'classes': list(filter(lambda x: is_important_for_class(x), sentences_supersenses))
    }
    
    return {
        bucket: [' '.join(df['token'].values) for df in bucketed_data[bucket]]
        for bucket in bucketed_data.keys()
    }