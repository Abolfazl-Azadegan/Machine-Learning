from fastai.text.all import *

# # Define dataset path
path = Path("C:/Users/EFE/.fastai/data/imdb")

# # Choose a tokenization method (default is WordTokenizer)
# tok = Tokenizer(tok=WordTokenizer())

# # Ensure tokenized directory exists
tok_path = path/'imdb_tok'
# tok_path.mkdir(exist_ok=True)

# for split in ['train', 'test', 'unsup']:
#     (tok_path/split).mkdir(exist_ok=True)  # Create train/test/unsup folders if missing

#     for file in (path/split).glob('*/*.txt'):  # Iterate over text files
#         txt = file.read_text()  # Read text
#         tokens = tok(txt)  # Tokenize
#         (tok_path/split/file.name).write_text(' '.join(tokens))  # Save tokenized text

# print("Tokenization completed successfully!")




# from collections import Counter

# # Load tokenized text
# all_tokens = []
# for file in (tok_path/'train').glob('*/*.txt'):  
#     all_tokens.extend(file.read_text().split())  # Read tokenized text

# # Create word frequency counter
# word_freq = Counter(all_tokens)

# # Save as pickle
# import pickle
# with open(tok_path/'counter.pkl', 'wb') as f:
#     pickle.dump(word_freq, f)

# print("✅ Vocabulary counter.pkl saved successfully!")


dls = TextDataLoaders.from_folder(tok_path, valid='test')

learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy)
learn.fine_tune(4, 1e-2)

# Save trained model
model_path = Path("C:/Users/EFE/.fastai/models")
learn.export(model_path / "imdb_reviewer.pkl")

print("✅ Model trained and saved successfully!")


