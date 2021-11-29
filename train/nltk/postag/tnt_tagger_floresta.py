import random

import joblib
import nltk
from json_database import JsonStorageXDG
from nltk.tag import tnt

from neon_modelhub import load_model

db = JsonStorageXDG("nltk_floresta_tnt_tagger", subfolder="ModelZoo/nltk")

MODEL_META = {
    "corpus": "floresta",
    "corpus_homepage": "http://www.linguateca.pt/Floresta",
    "lang": "pt",
    "model_id": "nltk_floresta_tnt_tagger",
    "tagset": "VISL (Portuguese)",
    "tagset_homepage": "https://visl.sdu.dk/visl/pt/symbolset-floresta.html",
    "algo": "TnT",
    "backoff_taggers": ["nltk_floresta_ngram_tagger"],
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
