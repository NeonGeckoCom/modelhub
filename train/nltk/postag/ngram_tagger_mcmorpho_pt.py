from random import shuffle

import joblib
import nltk
from json_database import JsonStorageXDG

db = JsonStorageXDG("nltk_macmorpho_ngram_tagger", subfolder="ModelZoo/nltk")
MODEL_META = {
    "corpus": "macmorpho",
    "model_id": "nltk_macmorpho_ngram_tagger",
    "corpus_homepage": "http://www.nilc.icmc.usp.br/macmorpho/",
    "tagset": "",
    "tagset_homepage": "http://www.nilc.icmc.usp.br/macmorpho/macmorpho-manual.pdf",
    "lang": "pt",
    "algo": "TrigramTagger",
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

regex_patterns = [
    (r"^[nN][ao]s?$", "ADP"),
    (r"^[dD][ao]s?$", "ADP"),
    (r"^[pP]el[ao]s?$", "ADP"),
    (r"^[nN]est[ae]s?$", "ADP"),
    (r"^[nN]um$", "ADP"),
    (r"^[nN]ess[ae]s?$", "ADP"),
    (r"^[nN]aquel[ae]s?$", "ADP"),
    (r"^\xe0$", "ADP"),
]

def_tagger = nltk.DefaultTagger('N')
affix_tagger = nltk.AffixTagger(
    train_data, backoff=def_tagger
)
unitagger = nltk.UnigramTagger(
    train_data, backoff=affix_tagger
)
rx_tagger = nltk.RegexpTagger(
    regex_patterns, backoff=unitagger
)
bitagger = nltk.BigramTagger(
    train_data, backoff=rx_tagger
)
tagger = nltk.TrigramTagger(
    train_data, backoff=bitagger
)
a = tagger.evaluate(test_data)

print("Accuracy of ngram tagger : ", a)  # 0.9260627517381154

db["accuracy"] = a
db.store()
joblib.dump(tagger, model_path)
