from random import shuffle

import joblib
import nltk
from json_database import JsonStorageXDG

from neon_classic_modelhub import load_model

db = JsonStorageXDG("nltk_macmorpho_brill_tagger", subfolder="ModelZoo/nltk")

MODEL_META = {
    "corpus": "macmorpho",
    "model_id": "nltk_macmorpho_brill_tagger",
    "corpus_homepage": "http://www.nilc.icmc.usp.br/macmorpho/",
    "tagset": "",
    "tagset_homepage": "http://www.nilc.icmc.usp.br/macmorpho/macmorpho-manual.pdf",
    "lang": "pt",
    "algo": "nltk.brill.fntbl37",
    "backoff_taggers": ["DefaultTagger", "AffixTagger", "UnigramTagger",
                        "RegexpTagger", "BigramTagger", "TrigramTagger"],
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

ngram_tagger = load_model(model_path.replace("brill", "ngram"))
tagger = nltk.BrillTaggerTrainer(ngram_tagger, nltk.brill.fntbl37())
tagger = tagger.train(train_data, max_rules=100)

a = tagger.evaluate(test_data)
db["accuracy"] = a
db.store()
joblib.dump(tagger, model_path)
