import random

import joblib
import nltk
from json_database import JsonStorageXDG
from nltk.corpus import treebank
from nltk.tag import DefaultTagger
from nltk.tag import tnt

db = JsonStorageXDG("nltk_treebank_tnt_tagger", subfolder="ModelZoo/nltk")

MODEL_META = {
    "corpus": "treebank",
    "lang": "en",
    "model_id": "nltk_treebank_tnt_tagger",
    "tagset": "Penn Treebank",
    "algo": "TnT",
    "backoff_taggers": ["DefaultTagger"],
    "required_packages": ["nltk"]
}
db.update(MODEL_META)
db.store()
model_path = db.path.replace(".json", ".pkl")

nltk.download('treebank')

corpus = list(treebank.tagged_sents())  # 3914
random.shuffle(corpus)
train_data = corpus[:3000]
test_data = corpus[3000:]

# initializing tagger
ngram_tagger = joblib.load(model_path.replace("tnt", "ngram"))

tagger = tnt.TnT(unk=ngram_tagger, Trained=True)

# training
tagger.train(train_data)

# evaluating
a = tagger.evaluate(test_data)

print("Accuracy of TnT Tagger : ", a)  # 0.892467083962875
db["accuracy"] = a
db.store()

joblib.dump(tagger, model_path)
