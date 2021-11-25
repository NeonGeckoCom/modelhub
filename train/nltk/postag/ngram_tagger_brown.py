import random

import joblib
import nltk
from json_database import JsonStorageXDG
from nltk import AffixTagger
from nltk.corpus import brown
from nltk.tag import UnigramTagger, BigramTagger, TrigramTagger

db = JsonStorageXDG("nltk_brown_ngram_tagger", subfolder="ModelZoo/nltk")
MODEL_META = {
    "corpus": "brown",
    "model_id": "nltk_brown_ngram_tagger",
    "tagset": "brown",
    "corpus_homepage": "http://www.hit.uib.no/icame/brown/bcm.html",
    "lang": "en",
    "algo": "TrigramTagger",
    "backoff_taggers": ["AffixTagger", "UnigramTagger", "BigramTagger",
                        "TrigramTagger"],
    "required_packages": ["nltk"]
}

db.update(MODEL_META)
db.store()
model_path = db.path.replace(".json", ".pkl")

# initializing training and testing set
nltk.download('brown')

corpus = [_ for _ in brown.tagged_sents()]  # 57340
random.shuffle(corpus)
cuttof = int(len(corpus) * 0.9)
train_data = corpus[:cuttof]
test_data = corpus[cuttof:]

# create tagger
affix = AffixTagger(train_data)
uni = UnigramTagger(train_data, backoff=affix)
bi = BigramTagger(train_data, backoff=uni)
tagger = TrigramTagger(train_data, backoff=bi)

a = tagger.evaluate(test_data)
print("Accuracy of ngram tagger : ", a)  # 0.9224974329959557

db["accuracy"] = a
db.store()
joblib.dump(tagger, model_path)
