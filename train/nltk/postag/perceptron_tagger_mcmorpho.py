from random import shuffle

import joblib
import nltk
from json_database import JsonStorageXDG
from nltk.tag import PerceptronTagger
db = JsonStorageXDG("nltk_macmorpho_perceptron_tagger", subfolder="ModelZoo/nltk")

MODEL_META = {
    "corpus": "macmorpho",
    "model_id": "nltk_macmorpho_perceptron_tagger",
    "corpus_homepage": "http://www.nilc.icmc.usp.br/macmorpho/",
    "tagset": "",
    "tagset_homepage": "http://www.nilc.icmc.usp.br/macmorpho/macmorpho-manual.pdf",
    "lang": "pt",
    "algo": "Perceptron",
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

shuffle(dataset)

cutoff = int(len(dataset) * 0.9)
train_data = dataset[:cutoff]
test_data = dataset[cutoff:]

tagger = PerceptronTagger(load=False)
tagger.train(train_data)

a = tagger.evaluate(test_data)

print("Accuracy of Perceptron tagger : ", a)  # 0.39968751319623325
db["accuracy"] = a
db.store()
joblib.dump(tagger, model_path)
