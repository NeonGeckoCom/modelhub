import random

import joblib
import nltk
from json_database import JsonStorageXDG
from nltk.corpus import treebank
from nltk.tag import tnt

from neon_modelhub import load_model

db = JsonStorageXDG("nltk_treebank_udep_tnt_tagger", subfolder="ModelZoo/nltk")

MODEL_META = {
    "corpus": "treebank",
    "lang": "en",
    "model_id": "nltk_treebank_udep_tnt_tagger",
    "tagset": "Universal Dependencies",
    "algo": "TnT",
    "backoff_taggers": ["nltk_treebank_udep_ngram_tagger"],
    "required_packages": ["nltk"]
}
db.update(MODEL_META)
db.store()
model_path = db.path.replace(".json", ".pkl")

nltk.download('universal_tagset')

corpus = list(treebank.tagged_sents(tagset='universal'))  # 3914
random.shuffle(corpus)
train_data = corpus[:3000]
test_data = corpus[3000:]

# initializing tagger
ngram_tagger = load_model(model_path.replace("tnt", "ngram"))

tagger = tnt.TnT(unk=ngram_tagger, Trained=True)

# training
tagger.train(train_data)

# evaluating
a = tagger.evaluate(test_data)

print("Accuracy of TnT Tagger : ", a)  # 0.892467083962875
db["accuracy"] = a
db.store()

joblib.dump(tagger, model_path)
