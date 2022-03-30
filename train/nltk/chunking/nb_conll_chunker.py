import random

import joblib
import nltk
from json_database import JsonStorageXDG
from nltk.chunk import tree2conlltags
from nltk.corpus import conll2000
from nltk.tag import ClassifierBasedTagger

from neon_classic_modelhub.chunkers.nltk_chunkers import ClassifierChunkParser
from neon_classic_modelhub.features.nltk_feats import NltkFeatures

db = JsonStorageXDG("nltk_conll2000_nb_chunker", subfolder="ModelZoo/nltk")
MODEL_META = {
    "corpus": "CONLL2000",
    "model_id": "nltk_conll2000_nb_chunker",
    "tagset": "conll_iob",
    "lang": "en",
    "algo": "NaiveBayes",
    "required_packages": ["nltk", "neon_classic_modelhub"]
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

# Transform the trees in IOB annotated triples [(word, pos, chunk), ...]
chunked_sents = [tree2conlltags(sent) for sent in train_sents]
chunked_test = [tree2conlltags(sent) for sent in test_sents]


# make it compatible with the tagger interface [((word, pos), chunk), ...]
def triplets2tagged_pairs(iob_sent):
    return [((word, pos), chunk) for word, pos, chunk in iob_sent]


train_data = [triplets2tagged_pairs(sent) for sent in chunked_sents]
test_data = [triplets2tagged_pairs(sent) for sent in chunked_test]

# train the tagger
tagger = ClassifierBasedTagger(
    train=train_data,
    feature_detector=NltkFeatures.extract_iob_features)

a = tagger.evaluate(test_data)

print("Accuracy of chunk tagger: ", a)  # 0.9101376235704594
joblib.dump(tagger, model_path)

# test the chunker
chunk = ClassifierChunkParser(model_id=MODEL_META["model_id"])
a = chunk.evaluate(test_sents)
print(a)
# ChunkParse score:
#     IOB Accuracy:  91.8%%
#     Precision:     85.3%%
#     Recall:        89.1%%
#     F-Measure:     87.2%%
