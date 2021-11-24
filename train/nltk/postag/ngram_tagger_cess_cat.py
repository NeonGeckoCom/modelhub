from random import shuffle

import joblib
import nltk
from json_database import JsonStorageXDG

db = JsonStorageXDG("nltk_cess_cat_ngram_tagger", subfolder="ModelZoo/nltk")
MODEL_META = {
    "corpus": "cess_cat",
    "corpus_homepage": "https://web.archive.org/web/20121023154634/http://clic.ub.edu/cessece/",
    "lang": "ca",
    "model_id": "nltk_cess_cat_ngram_tagger",
    "tagset": "EAGLES",
    "tagset_homepage": "http://www.ilc.cnr.it/EAGLES96/annotate/annotate.html",
    "backoff_taggers": ["AffixTagger", "UnigramTagger",
                        "BigramTagger", "TrigramTagger"],
    "algo": "TrigramTagger",
    "required_packages": ["nltk"]
}
db.update(MODEL_META)
db.store()
model_path = db.path.replace(".json", ".pkl")
print(model_path)
# EAGLES
# http://www.ilc.cnr.it/EAGLES96/annotate/annotate.html
nltk.download('cess_cat')

corpus = [sent for sent in nltk.corpus.cess_cat.tagged_sents()]
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
