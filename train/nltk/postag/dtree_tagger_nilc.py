import random

import joblib
from json_database import JsonStorageXDG
from nltk.tag.sequential import ClassifierBasedPOSTagger

import biblioteca
from biblioteca.corpora.external import NILC
from nltk.classify import DecisionTreeClassifier
db = JsonStorageXDG("nltk_nilc_dtree_tagger", subfolder="ModelZoo/nltk")
MODEL_META = {
    "corpus": "NILC_taggers",
    "corpus_homepage": "http://www.nilc.icmc.usp.br/nilc/tools/nilctaggers.html",
    "model_id": "nltk_nilc_dtree_tagger",
    "tagset": "NILC",
    "tagset_homepage": "http://www.nilc.icmc.usp.br/nilc/download/tagsetcompleto.doc",
    "lang": "pt-br",
    "algo": "DecisionTreeClassifier",
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

tagger = ClassifierBasedPOSTagger(
    train=train_data,
    classifier_builder=DecisionTreeClassifier.train)

a = tagger.evaluate(test_data)

print("Accuracy:", a)  # 0.8753341915041093
db["accuracy"] = a
db.store()
joblib.dump(tagger, model_path)
