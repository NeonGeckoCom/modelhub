import random

import joblib
from json_database import JsonStorageXDG
from nltk.corpus import treebank
from nltk.tag import DefaultTagger, AffixTagger
from nltk.tag import UnigramTagger, BigramTagger, TrigramTagger
from nltk.tag import DefaultTagger
from nltk.tag import RegexpTagger

db = JsonStorageXDG("nltk_treebank_ngram_tagger", subfolder="ModelZoo/nltk")
MODEL_META = {
    "corpus": "treebank",
    "lang": "en",
    "model_id": "nltk_treebank_ngram_tagger",
    "tagset": "Penn Treebank",
    "algo": "TrigramTagger",
    "backoff_taggers": ["DefaultTagger", "UnigramTagger", "BigramTagger",
                        "TrigramTagger"],
    "required_packages": ["nltk"]
}
db.update(MODEL_META)
db.store()
model_path = db.path.replace(".json", ".pkl")

corpus = list(treebank.tagged_sents())  # 3914
random.shuffle(corpus)
train_data = corpus[:3000]
test_data = corpus[3000:]

# create tagger
patterns = [
    (r'^\d+$', 'CD'),
    (r'.*ing$', 'VBG'),  # gerunds, i.e. wondering
    (r'.*ment$', 'NN'),  # i.e. wonderment
    (r'.*ful$', 'JJ')  # i.e. wonderful
]
affix = AffixTagger(train_data, backoff=DefaultTagger('NN'))
rx = RegexpTagger(patterns, backoff=affix)
uni = UnigramTagger(train_data, backoff=rx)
bi = BigramTagger(train_data, backoff=uni)
tagger = TrigramTagger(train_data, backoff=bi)

a = tagger.evaluate(test_data)

print("Accuracy of Ngram tagger : ", a)  # 0.8806388948845241
db["accuracy"] = a
db.store()
joblib.dump(tagger, model_path)
