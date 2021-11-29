from random import shuffle

import joblib
import nltk
from json_database import JsonStorageXDG

import biblioteca

biblioteca.download("cess_esp_udep")
cess = biblioteca.load_corpus("cess_esp_udep")

db = JsonStorageXDG("nltk_cess_esp_udep_ngram_tagger", subfolder="ModelZoo/nltk")
MODEL_META = {
    "corpus": "cess_esp_udep",
    "corpus_homepage": "https://web.archive.org/web/20121023154634/http://clic.ub.edu/cessece/",
    "lang": "es",
    "model_id": "nltk_cess_esp_udep_ngram_tagger",
    "algo": "TrigramTagger",
    "tagset": "Universal Dependencies",
    "backoff_taggers": ["AffixTagger", "UnigramTagger",
                        "BigramTagger", "TrigramTagger"],
    "required_packages": ["nltk"]
}
db.update(MODEL_META)
db.store()
model_path = db.path.replace(".json", ".pkl")

corpus = list(cess.tagged_sentences())
shuffle(corpus)
cutoff = int(len(corpus) * 0.9)
train_data = corpus[:cutoff]
test_data = corpus[cutoff:]

affix_tagger = nltk.AffixTagger(
    train_data
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

print("Accuracy of ngram tagger : ", a)  # 0.9350522453537529
db["accuracy"] = a
db.store()

joblib.dump(tagger, model_path)
