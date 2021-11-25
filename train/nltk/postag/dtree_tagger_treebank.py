import random

import joblib
import nltk
from json_database import JsonStorageXDG
from nltk.corpus import treebank
from nltk.tag.sequential import ClassifierBasedPOSTagger
from nltk.classify import DecisionTreeClassifier

db = JsonStorageXDG("nltk_treebank_dtree_tagger", subfolder="ModelZoo/nltk")
MODEL_META = {
    "corpus": "treebank",
    "lang": "en",
    "model_id": "nltk_treebank_dtree_tagger",
    "tagset": "Penn Treebank",
    "algo": "DecisionTreeClassifier",
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
    train=train_data,
    classifier_builder=DecisionTreeClassifier.train)

a = tagger.evaluate(test_data)

print("Accuracy: ", a)
db["accuracy"] = a
db.store()
joblib.dump(tagger, model_path)
