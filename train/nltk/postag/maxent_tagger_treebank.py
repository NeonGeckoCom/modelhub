import random

import joblib
import nltk
from json_database import JsonStorageXDG
from nltk.classify import MaxentClassifier
from nltk.corpus import treebank
from nltk.tag.sequential import ClassifierBasedPOSTagger

db = JsonStorageXDG("nltk_treebank_maxent_tagger", subfolder="ModelZoo/nltk")

MODEL_META = {
    "corpus": "treebank",
    "lang": "en",
    "model_id": "nltk_treebank_maxent_tagger",
    "tagset": "Penn Treebank",
    "algo": "MaxentClassifier",
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

tagger = ClassifierBasedPOSTagger(
    train=train_data, classifier_builder=MaxentClassifier.train)

a = tagger.evaluate(test_data)

print("Maxent Accuracy : ", a)  # 0.9258363911072739
db["accuracy"] = a
db.store()

joblib.dump(tagger, model_path)
