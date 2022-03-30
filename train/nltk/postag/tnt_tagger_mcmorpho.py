import random

import joblib
import nltk
from json_database import JsonStorageXDG
from nltk.tag import tnt

from neon_classic_modelhub import load_model

db = JsonStorageXDG("nltk_macmorpho_tnt_tagger", subfolder="ModelZoo/nltk")

MODEL_META = {
    "corpus": "macmorpho",
    "corpus_homepage": "http://www.linguateca.pt/Floresta",
    "lang": "pt",
    "model_id": "nltk_macmorpho_tnt_tagger",
    "tagset": "",
    "tagset_homepage": "http://www.nilc.icmc.usp.br/macmorpho/macmorpho-manual.pdf",
    "algo": "TnT",
    "backoff_taggers": ["nltk_macmorpho_ngram_tagger"],
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

# initializing tagger
ngram_tagger = load_model(model_path.replace("tnt", "ngram"))

tagger = tnt.TnT(unk=ngram_tagger, Trained=True)

# training
tagger.train(train_data)

# evaluating
a = tagger.evaluate(test_data)

print("Accuracy of TnT Tagger : ", a)  # 0.892467083962875
db["accuracy"] = a
db.store()

joblib.dump(tagger, model_path)
