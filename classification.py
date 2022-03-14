import pickle
import numpy as np
import pandas as pd

class Classifier:
    def __init__(self, tokens, retrieve_text):
        self.tokens = tokens
        self.retrieve_text = retrieve_text

        with open('data/model.pkl', 'rb') as f:
            self.crf = pickle.load(f)

    def word2features(self, sent, i):
        word = sent[i][0]
        postag = sent[i][3]
        
        features = {
            'word': word,
            'word[-3:]': word[-3:],
            'word[-2:]': word[-2:],
            'postag': postag,
            'postag[:2]': postag[:2],
            'lemma': sent[i][1],
            'general_pos': sent[i][2],
            'dependency': sent[i][4],
            'supersense': sent[i][5],
            'entity': sent[i][6],
            'entity_type': sent[i][7],
            'entity_category': sent[i][8]
        }
        if i > 0:
            word1 = sent[i-1][0]
            postag1 = sent[i-1][2]
            features.update({
                '-1:postag': postag1,
                '-1:postag[:2]': postag1[:2],
            })
        else:
            features['BOS'] = True

        if i < len(sent)-1:
            word1 = sent[i+1][0]
            postag1 = sent[i+1][2]
            features.update({
                '+1:postag': postag1,
                '+1:postag[:2]': postag1[:2],
            })
        else:
            features['EOS'] = True

        return features

    def sent2features(self, sent):
        return [self.word2features(sent, i) for i in range(len(sent))]

    def preprocess(self, input_text):
        def agg_func(s): return [(w, l, p, fp, dr, sc, e, et, ec) for w, l, p, fp, dr, sc, e, et, ec in zip(
            s['word'].values.tolist(),
            s['lemma'].values.tolist(),
            s['POS_tag'].values.tolist(),
            s['fine_POS_tag'].values.tolist(),
            s['dependency_relation'].values.tolist(),
            s['supersense_category'].values.tolist(),
            s['entity'].values.tolist(),
            s['entity_type'].values.tolist(),
            s['entity_category'].values.tolist(),
        )]
        
        raw_features = [agg_func(df) for df in input_text]
        sentences = [s for s in raw_features]
        prediction_input = np.array([self.sent2features(s) for s in sentences], dtype=object)

        return prediction_input
    
    def get_metadata(self, df):
        # Only keep all the metadata rows
        exclude_mask = df['pred'] == 'O'
        metadata_df = df[~exclude_mask]

        # Make template for returning values
        metadata = {
            'class': [],
            'attr': [],
        }

        # Extract the words for all metadata
        last_metadata = None
        for index, row in metadata_df.iterrows():
            # Split prediction into the iob tag and the actual prediction
            iob, metadata_type = tuple(row['pred'].split('-'))

            if last_metadata:
                if iob == 'B':
                    # New metadata entity starts
                    # Move last metadata to metadata object
                    metadata[last_metadata[1]].append(' '.join(last_metadata[0]))
                    
                    # Initiate new entity
                    last_metadata = [[row['word']], metadata_type]
                else:
                    # Append to existing entity
                    last_metadata[0] = last_metadata[0] + [row['word']]
            else:
                # Initiate new entity
                last_metadata = [[row['word']], metadata_type]
        
        if last_metadata:
            # "Empty" metadata
            metadata[last_metadata[1]].append(' '.join(last_metadata[0]))
        
        return metadata
 
    def get_prediction(self):
        input_text = self.retrieve_text()['class']
        if input_text:
            prediction_input = self.preprocess(input_text)
            y_pred = self.crf.predict(prediction_input)

            for i in range(len(input_text)):
                input_text[i]['pred'] = y_pred[i]
            
            return self.get_metadata(pd.concat(input_text))
        else:
            raise Exception(
                'Please create an entity-summarized input text and a bucketed output first before invoking this function.')
        

