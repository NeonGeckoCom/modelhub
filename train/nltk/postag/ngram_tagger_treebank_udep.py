import random
import nltk
import joblib
from json_database import JsonStorageXDG
from nltk.corpus import treebank
from nltk.tag import AffixTagger
from nltk.tag import DefaultTagger
from nltk.tag import RegexpTagger
from nltk.tag import UnigramTagger, BigramTagger, TrigramTagger

db = JsonStorageXDG("nltk_treebank_udep_ngram_tagger", subfolder="ModelZoo/nltk")
MODEL_META = {
    "corpus": "treebank",
    "lang": "en",
    "model_id": "nltk_treebank_udep_ngram_tagger",
    "tagset": "Universal Dependencies",
    "algo": "TrigramTagger",
    "backoff_taggers": ["DefaultTagger", "UnigramTagger", "BigramTagger",
                        "TrigramTagger"],
    "required_packages": ["nltk"]
}
db.update(MODEL_META)
db.store()
model_path = db.path.replace(".json", ".pkl")

nltk.download('universal_tagset')

corpus = list(treebank.tagged_sents(tagset='universal'))  # 3914
random.shuffle(corpus)
train_data = corpus[:3000]
test_data = corpus[3000:]

# create tagger
affix = AffixTagger(train_data, backoff=DefaultTagger('NN'))
uni = UnigramTagger(train_data, backoff=affix)
bi = BigramTagger(train_data, backoff=uni)
tagger = TrigramTagger(train_data, backoff=bi)

a = tagger.evaluate(test_data)

print("Accuracy of Ngram tagger : ", a)  # 0.8806388948845241
db["accuracy"] = a
db.store()
joblib.dump(tagger, model_path)
