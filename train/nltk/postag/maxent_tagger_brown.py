import random

import joblib
import nltk
from json_database import JsonStorageXDG
from nltk.classify import MaxentClassifier
from nltk.corpus import brown
from nltk.tag.sequential import ClassifierBasedPOSTagger

db = JsonStorageXDG("nltk_brown_maxent_tagger", subfolder="ModelZoo/nltk")
MODEL_META = {
    "corpus": "brown",
    "corpus_homepage": "http://www.hit.uib.no/icame/brown/bcm.html",
    "lang": "en",
    "model_id": "nltk_brown_maxent_tagger",
    "tagset": "brown",
    "algo": "ClassifierBasedPOSTagger",
    "classifier": "MaxentClassifier",
    "required_packages": ["nltk"]
}
db.update(MODEL_META)
db.store()
model_path = db.path.replace(".json", ".pkl")

# initializing training and testing set
nltk.download('brown')

corpus = list(brown.tagged_sents())  # 3914
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
