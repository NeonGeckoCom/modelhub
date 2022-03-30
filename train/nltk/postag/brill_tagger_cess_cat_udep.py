from random import shuffle

import joblib
import nltk
from json_database import JsonStorageXDG

import biblioteca
from neon_classic_modelhub import load_model

biblioteca.download("cess_cat_udep")
cess = biblioteca.load_corpus("cess_cat_udep")

db = JsonStorageXDG("nltk_cess_cat_udep_brill_tagger", subfolder="ModelZoo/nltk")
MODEL_META = {
    "corpus": "cess_cat_udep",
    "corpus_homepage": "https://github.com/OpenJarbas/biblioteca",
    "lang": "ca",
    "model_id": "nltk_cess_cat_udep_brill_tagger",
    "tagset": "Universal Dependencies",
    "algo": "nltk.brill.fntbl37",
    "backoff_taggers": ["DefaultTagger", "AffixTagger", "UnigramTagger",
                        "BigramTagger", "TrigramTagger"],
    "required_packages": ["nltk"],
    "train/test": "80/20"
}
db.update(MODEL_META)
db.store()
model_path = db.path.replace(".json", ".pkl")

corpus = list(cess.tagged_sentences())
shuffle(corpus)
cutoff = int(len(corpus) * 0.8)
train_data = corpus[:cutoff]
test_data = corpus[cutoff:]

ngram_tagger = load_model(model_path.replace("brill", "ngram"))

tagger = nltk.BrillTaggerTrainer(ngram_tagger, nltk.brill.fntbl37())
tagger = tagger.train(train_data)

a = tagger.evaluate(test_data)

print("Accuracy of Brill tagger : ", a)  # 0.9745613865781397
MODEL_META["accuracy"] = a
db.store()

joblib.dump(tagger, model_path)
