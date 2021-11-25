import random

import joblib
import nltk
from json_database import JsonStorageXDG
from nltk import TrigramTagger, BigramTagger, UnigramTagger
from nltk.chunk import tree2conlltags
from nltk.corpus import conll2000

from neon_modelhub.chunkers.nltk_chunkers import PostagChunkParser

db = JsonStorageXDG("nltk_conll2000_ngram_ptchunker", subfolder="ModelZoo/nltk")
MODEL_META = {
    "corpus": "CONLL2000",
    "model_id": "nltk_conll2000_ngram_ptchunker",
    "tagset": "",
    "lang": "en",
    "algo": "TrigramTagger",
    "backoff_taggers": ["UnigramTagger", "BigramTagger", "TrigramTagger"],
    "required_packages": ["nltk", "neon_modelhub"]
}
db.update(MODEL_META)
db.store()
model_path = db.path.replace(".json", ".pkl")

# initializing training and testing set
nltk.download('conll2000')
shuffled_conll_sents = list(conll2000.chunked_sents())
random.shuffle(shuffled_conll_sents)
train_sents = shuffled_conll_sents[:int(len(shuffled_conll_sents) * 0.9)]
test_sents = shuffled_conll_sents[int(len(shuffled_conll_sents) * 0.9 + 1):]

# Extract only the (POS-TAG, IOB-CHUNK-TAG) pairs
train_data = [
    [(pos_tag, chunk_tag) for word, pos_tag, chunk_tag in tree2conlltags(sent)]
    for sent in train_sents]
test_data = [
    [(pos_tag, chunk_tag) for word, pos_tag, chunk_tag in tree2conlltags(sent)]
    for sent in test_sents]

# Train a NgramTagger
t1 = UnigramTagger(train_data)
t2 = BigramTagger(train_data, backoff=t1)
tagger = TrigramTagger(train_data, backoff=t2)
a = tagger.evaluate(test_data)

print("Accuracy of chunk tagger: ", a)  # 0.8873818489203105

joblib.dump(tagger, model_path)

chunk = PostagChunkParser(model_id=MODEL_META["model_id"])

a = chunk.evaluate(test_sents)
print(a)
# ChunkParse score:
#     IOB Accuracy:  88.6%%
#     Precision:     80.6%%
#     Recall:        85.6%%
#     F-Measure:     83.1%%
