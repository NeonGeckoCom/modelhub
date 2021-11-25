import random

import joblib
import nltk
from json_database import JsonStorageXDG
from nltk.corpus import brown
from nltk.tag import tnt

from neon_modelhub import load_model

db = JsonStorageXDG("nltk_brown_tnt_tagger", subfolder="ModelZoo/nltk")
MODEL_META = {
    "corpus": "brown",
    "lang": "en",
    "corpus_homepage": "http://www.hit.uib.no/icame/brown/bcm.html",
    "model_id": "nltk_brown_tnt_tagger",
    "tagset": "brown",
    "algo": "TnT",
    "required_packages": ["nltk"]
}
db.update(MODEL_META)
db.store()
model_path = db.path.replace(".json", ".pkl")

# initializing training and testing set
nltk.download('brown')

corpus = brown.tagged_sents()  # 3914
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

print("Accuracy: ", a)
db["accuracy"] = a
db.store()
joblib.dump(tagger, model_path)
