import random

import joblib
import nltk
from json_database import JsonStorageXDG
from nltk.classify import DecisionTreeClassifier
from nltk.corpus import treebank
from nltk.tag.sequential import ClassifierBasedPOSTagger

db = JsonStorageXDG("nltk_treebank_udep_dtree_tagger", subfolder="ModelZoo/nltk")
MODEL_META = {
    "corpus": "treebank",
    "lang": "en",
    "model_id": "nltk_treebank_udep_dtree_tagger",
    "tagset": "Universal Dependencies",
    "algo": "DecisionTreeClassifier",
    "required_packages": ["nltk"]
}
db.update(MODEL_META)
db.store()
model_path = db.path.replace(".json", ".pkl")

nltk.download('treebank')
nltk.download('universal_tagset')

corpus = list(treebank.tagged_sents(tagset='universal'))  # 3914
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
