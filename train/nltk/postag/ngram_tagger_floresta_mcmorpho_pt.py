from random import shuffle
from string import punctuation

import joblib
import nltk
from json_database import JsonStorageXDG

db = JsonStorageXDG("nltk_floresta_macmorpho_ngram_tagger", subfolder="ModelZoo/nltk")
MODEL_META = {
    "corpus": "floresta + macmorpho",
    "lang": "pt",
    "model_id": "nltk_floresta_macmorpho_ngram_tagger",
    "corpus_homepage": "https://www.nltk.org/nltk_data",
    "tagset": "Universal Dependencies",
    "algo": "TrigramTagger",
    "backoff_taggers": ["DefaultTagger", "AffixTagger", "UnigramTagger",
                        "RegexpTagger", "BigramTagger", "TrigramTagger"],
    "required_packages": ["nltk"]
}
db.update(MODEL_META)
db.store()
model_path = db.path.replace(".json", ".pkl")

nltk.download('mac_morpho')
nltk.download('floresta')


def convert_to_universal_tag(t, reverse=False):
    tagdict = {
        'n': "NOUN",
        'num': "NUM",
        'v-fin': "VERB",
        'v-inf': "VERB",
        'v-ger': "VERB",
        'v-pcp': "VERB",
        'pron-det': "PRON",
        'pron-indp': "PRON",
        'pron-pers': "PRON",
        'art': "DET",
        'adv': "ADV",
        'conj-s': "CONJ",
        'conj-c': "CONJ",
        'conj-p': "CONJ",
        'adj': "ADJ",
        'ec': "PRT",
        'pp': "ADP",
        'prp': "ADP",
        'prop': "NOUN",
        'pro-ks-rel': "PRON",
        'proadj': "PRON",
        'prep': "ADP",
        'nprop': "NOUN",
        'vaux': "VERB",
        'propess': "PRON",
        'v': "VERB",
        'vp': "VERB",
        'in': "X",
        'prp-': "ADP",
        'adv-ks': "ADV",
        'dad': "NUM",
        'prosub': "PRON",
        'tel': "NUM",
        'ap': "NUM",
        'est': "NOUN",
        'cur': "X",
        'pcp': "VERB",
        'pro-ks': "PRON",
        'hor': "NUM",
        'pden': "ADV",
        'dat': "NUM",
        'kc': "ADP",
        'ks': "ADP",
        'adv-ks-rel': "ADV",
        'npro': "NOUN",
    }
    if t in ["N|AP", "N|DAD", "N|DAT", "N|HOR", "N|TEL"]:
        t = "NUM"
    if reverse:
        if "|" in t: t = t.split("|")[0]
    else:
        if "+" in t: t = t.split("+")[1]
        if "|" in t: t = t.split("|")[1]
        if "#" in t: t = t.split("#")[0]
    t = t.lower()
    return tagdict.get(t, "." if all(tt in punctuation for tt in t) else t)


mac_morpho = [
    [(w, convert_to_universal_tag(t, reverse=True)) for (w, t) in sent]
    for sent in nltk.corpus.mac_morpho.tagged_sents()]

floresta = [[(w, convert_to_universal_tag(t)) for (w, t) in sent]
            for sent in nltk.corpus.floresta.tagged_sents()]

dataset = floresta + mac_morpho
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

def_tagger = nltk.DefaultTagger('NOUN')
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

print("Accuracy of ngram tagger : ", a)  # 0.941382754573093
db["accuracy"] = a
db.store()
joblib.dump(tagger, model_path)
