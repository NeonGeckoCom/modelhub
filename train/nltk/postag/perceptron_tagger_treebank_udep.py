import random

import joblib
from json_database import JsonStorageXDG
from nltk.corpus import treebank
from nltk.tag import PerceptronTagger
import nltk


db = JsonStorageXDG("nltk_treebank_udep_perceptron_tagger", subfolder="ModelZoo/nltk")
MODEL_META = {
    "corpus": "treebank",
    "model_id": "nltk_treebank_udep_perceptron_tagger",
    "tagset": "Universal Dependencies",
    "lang": "en",
    "algo": "Perceptron",
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

tagger = PerceptronTagger(load=False)
tagger.train(train_data)
a = tagger.evaluate(test_data)

print("Accuracy of Perceptron tagger : ", a)  # 0.39968751319623325
db["accuracy"] = a
db.store()

joblib.dump(tagger, model_path)
