import pickle
import numpy as np
import pandas as pd
import json

class Classifier:
    def __init__(self, tokens, retrieve_text):
        self.tokens = tokens
        self.retrieve_text = retrieve_text
        self.columns = ['word', 'lemma', 'POS_tag', 'fine_POS_tag', 'dependency_relation', 'event', 'supersense_category', 'entity', 'entity_type', 'entity_category', 'total_occurences', 'class_occurences', 'attribute_occurences']

        with open('models/model.pkl', 'rb') as f:
            self.crf = pickle.load(f)
        
        with open('models/fasttext-model.pkl', 'rb') as f:
            self.fasttext = pickle.load(f)
        
        with open('../data/genmymodel/genmymodel_uml_extracted_metadata_final.json') as json_file:
            gmm_data = json.load(json_file)

        # Store all classes and attributes independent of eachother
        all_classes = []
        all_attrs = []

        # Loop over all metadata and append to proper list
        for file, metadata in gmm_data.items():
            if 'classes' in metadata.keys():
                all_classes.append(metadata['classes'])

            if 'attributes' in metadata.keys():
                all_attrs.append(metadata['attributes'])

        flatten = lambda t: [item for sublist in t for item in sublist]

        self.all_classes = flatten(all_classes)
        self.all_attrs = flatten(all_attrs)

    def word2features(self, sent, i):
        word = sent[i][1]
        postag = sent[i][3]
        fine_postag = sent[i][4]
        
        features = {
            label: data
            for label, data in zip(self.columns, sent[i])
        }
        
        features.update({
            'word.lower()': word.lower(),
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(),
            'word[-3:]': word[-3:],
            'word[-2:]': word[-2:],
            'postag[:2]': postag[:2],
            'postag[:2]': postag[:2],
            'finepostag[:2]': fine_postag[:2],
            'finepostag[:2]': fine_postag[:2],
        })
        if i > 0:
            word1 = sent[i-1][1]
            postag1 = sent[i-1][3]
            finepostag1 = sent[i-1][4]
            features.update({
                '-1:word.lower()': word1.lower(),
                '-1:word.istitle()': word1.istitle(),
                '-1:word.isupper()': word1.isupper(),
                '-1:postag': postag1,
                '-1:postag[:2]': postag1[:2],
                '-1:finepostag': finepostag1,
                '-1:finepostag[:2]': finepostag1[:2],
            })
        else:
            features['BOS'] = True

        if i < len(sent)-1:
            word1 = sent[i+1][1]
            postag1 = sent[i+1][3]
            finepostag1 = sent[i-1][4]
            features.update({
                '+1:word.lower()': word1.lower(),
                '+1:word.istitle()': word1.istitle(),
                '+1:word.isupper()': word1.isupper(),
                '+1:postag': postag1,
                '+1:postag[:2]': postag1[:2],
                '+1:finepostag': finepostag1,
                '+1:finepostag[:2]': finepostag1[:2],
            })
        else:
            features['EOS'] = True

        word_embedding = self.fasttext.wv.get_vector(word)
        
        features.update({
            f'emb_pos_{i}': word_embedding[i]
            for i in range(len(word_embedding))
        })

        return features

    def sent2features(self, sent):
        return [self.word2features(sent, i) for i in range(len(sent))]

    def preprocess(self, input_text):
        def get_gmm_data(df):
            df['total_occurences'] = 0
            df['class_occurences'] = 0
            df['attribute_occurences'] = 0

            noungroup = []
            noungroup_indices = []

            for index, row in df.iterrows():
                if isinstance(row['fine_POS_tag'], str) and row['fine_POS_tag'][:2] == 'NN':
                    noungroup.append(row['word'])
                    noungroup_indices.append(index)
                else:
                    if len(noungroup) == 0:
                        continue
                    else:
                        full_ng = ' '.join(noungroup).lower()
                        attr_no = self.all_attrs.count(full_ng)
                        class_no = self.all_classes.count(full_ng)
                        
                        for noun_index in noungroup_indices:
                            df.loc[noun_index, ['class_occurences', 'attribute_occurences', 'total_occurences']] = [class_no, attr_no, attr_no + class_no]
                            
                        noungroup = []
                        noungroup_indices = []
            
            return df

        agg_func = lambda s: list(map(lambda w: tuple(w), get_gmm_data(s).values.tolist()))
        
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
        

