from random import shuffle

import joblib
import nltk
from json_database import JsonStorageXDG

import biblioteca

biblioteca.download("cess_cat_udep")
cess = biblioteca.load_corpus("cess_cat_udep")

db = JsonStorageXDG("nltk_cess_cat_udep_ngram_tagger", subfolder="ModelZoo/nltk")
MODEL_META = {
    "corpus": "cess_cat_udep",
    "corpus_homepage": "https://github.com/OpenJarbas/biblioteca",
    "lang": "ca",
    "model_id": "nltk_cess_cat_udep_ngram_tagger",
    "tagset": "Universal Dependencies",
    "algo": "TrigramTagger",
    "backoff_taggers": ["AffixTagger", "UnigramTagger", "BigramTagger",
                        "TrigramTagger"],
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

def_tagger = nltk.DefaultTagger('NOUN')
affix_tagger = nltk.AffixTagger(
    train_data, backoff=def_tagger
)
unitagger = nltk.UnigramTagger(
    train_data, backoff=affix_tagger
)
bitagger = nltk.BigramTagger(
    train_data, backoff=unitagger
)
tagger = nltk.TrigramTagger(
    train_data, backoff=bitagger
)

a = tagger.evaluate(test_data)

db["accuracy"] = a
print("Accuracy of ngram tagger : ", a)  # 0.9666686290670748
db.store()

joblib.dump(tagger, model_path)
