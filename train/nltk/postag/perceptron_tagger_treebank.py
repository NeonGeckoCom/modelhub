import random

import joblib
from json_database import JsonStorageXDG
from nltk.corpus import treebank
from nltk.tag import PerceptronTagger

db = JsonStorageXDG("nltk_treebank_perceptron_tagger", subfolder="ModelZoo/nltk")
MODEL_META = {
    "corpus": "treebank",
    "corpus_homepage": "http://www.hit.uib.no/icame/brown/bcm.html",
    "model_id": "nltk_treebank_perceptron_tagger",
    "tagset": "Penn Treebank",
    "lang": "en",
    "algo": "Perceptron",
    "required_packages": ["nltk"]
}
db.update(MODEL_META)
db.store()
model_path = db.path.replace(".json", ".pkl")

corpus = list(treebank.tagged_sents())  # 3914
random.shuffle(corpus)
train_data = corpus[:3000]
test_data = corpus[3000:]

tagger = PerceptronTagger(load=False)
tagger.train(train_data)
a = tagger.evaluate(test_data)

print("Accuracy of Perceptron tagger : ", a)  # 0.39968751319623325
db["accuracy"] = a
db.store()

joblib.dump(tagger, model_path)
