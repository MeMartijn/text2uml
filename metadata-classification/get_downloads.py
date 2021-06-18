from flair.embeddings import TransformerWordEmbeddings

for embedding in ['bert-base-uncased', 'EleutherAI/gpt-neo-1.3B', 'transfo-xl-wt103', 'xlnet-base-cased', 'XLM xlm-mlm-en-2048']:
    print(f'Downloading {embedding}')
    TransformerWordEmbeddings(embedding)
    print(f'Finished downloading {embedding}')

