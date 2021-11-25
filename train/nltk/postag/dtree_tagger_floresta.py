import random

import joblib
import nltk
from json_database import JsonStorageXDG
from nltk.tag.sequential import ClassifierBasedPOSTagger
from nltk.classify import DecisionTreeClassifier
db = JsonStorageXDG("nltk_floresta_dtree_tagger", subfolder="ModelZoo/nltk")
MODEL_META = {
    "corpus": "floresta",
    "corpus_homepage": "http://www.linguateca.pt/Floresta",
    "lang": "pt",
    "model_id": "nltk_floresta_dtree_tagger",
     "tagset": "VISL (Portuguese)",
    "tagset_homepage": "https://visl.sdu.dk/visl/pt/symbolset-floresta.html",
    "algo": "DecisionTreeClassifier",
    "required_packages": ["nltk"]
}
db.update(MODEL_META)
db.store()
model_path = db.path.replace(".json", ".pkl")


nltk.download('floresta')


def clean_tag(t):
    if "+" in t: t = t.split("+")[1]
    if "|" in t: t = t.split("|")[1]
    if "#" in t: t = t.split("#")[0]
    t = t.lower()
    return t


floresta = [[(w, clean_tag(t)) for (w, t) in sent]
            for sent in nltk.corpus.floresta.tagged_sents()]
random.shuffle(floresta)

cutoff = int(len(floresta) * 0.9)
train_data = floresta[:cutoff]
test_data = floresta[cutoff:]


tagger = ClassifierBasedPOSTagger(
    train=train_data,
    classifier_builder=DecisionTreeClassifier.train)

a = tagger.evaluate(test_data)

print("Accuracy: ", a)
db["accuracy"] = a
db.store()
joblib.dump(tagger, model_path)
