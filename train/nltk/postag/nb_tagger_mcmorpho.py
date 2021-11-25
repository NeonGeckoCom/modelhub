import random

import joblib
import nltk
from json_database import JsonStorageXDG
from nltk.corpus import brown
from nltk.tag.sequential import ClassifierBasedPOSTagger

db = JsonStorageXDG("nltk_macmorpho_nb_tagger", subfolder="ModelZoo/nltk")
MODEL_META = {
    "corpus": "macmorpho",
    "corpus_homepage": "http://www.nilc.icmc.usp.br/macmorpho/",
    "tagset": "",
    "tagset_homepage": "http://www.nilc.icmc.usp.br/macmorpho/macmorpho-manual.pdf",
    "lang": "pt",
    "model_id": "nltk_macmorpho_nb_tagger",
    "algo": "NaiveBayes",
    "required_packages": ["nltk"]
}
db.update(MODEL_META)
db.store()
model_path = db.path.replace(".json", ".pkl")


nltk.download('mac_morpho')


def clean_tag(t, ):
    if "|" in t:
        t = t.split("|")[0]
    return t


dataset = [[(w, clean_tag(t)) for (w, t) in sent]
           for sent in nltk.corpus.mac_morpho.tagged_sents()]

random.shuffle(dataset)

cutoff = int(len(dataset) * 0.9)
train_data = dataset[:cutoff]
test_data = dataset[cutoff:]


tagger = ClassifierBasedPOSTagger(train=train_data)

a = tagger.evaluate(test_data)

print("Accuracy: ", a)
db["accuracy"] = a
db.store()
joblib.dump(tagger, model_path)
