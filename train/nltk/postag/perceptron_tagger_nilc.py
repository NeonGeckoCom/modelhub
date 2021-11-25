import random

import joblib
from json_database import JsonStorageXDG
from nltk.tag import PerceptronTagger

import biblioteca
from biblioteca.corpora.external import NILC

db = JsonStorageXDG("nltk_nilc_perceptron_tagger", subfolder="ModelZoo/nltk")
MODEL_META = {
    "corpus": "NILC_taggers",
    "corpus_homepage": "http://www.nilc.icmc.usp.br/nilc/tools/nilctaggers.html",
    "model_id": "nltk_nilc_perceptron_tagger",
    "tagset": "NILC",
    "tagset_homepage": "http://www.nilc.icmc.usp.br/nilc/download/tagsetcompleto.doc",
    "lang": "pt-br",
    "algo": "Perceptron",
    "required_packages": ["nltk"]
}

db.update(MODEL_META)
db.store()
model_path = db.path.replace(".json", ".pkl")

biblioteca.download("NILC_taggers")
nilc = NILC()

data = [s for s in nilc.tagged_sentences()]
random.shuffle(data)
cutoff = int(len(data) * 0.9)
train_data = data[:cutoff]
test_data = data[cutoff:]

tagger = PerceptronTagger(load=False)
tagger.train(train_data)

a = tagger.evaluate(test_data)

print("Accuracy of Perceptron tagger : ", a)  # 0.39968751319623325
db["accuracy"] = a
db.store()
joblib.dump(tagger, model_path)
