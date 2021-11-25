import random

import joblib
import nltk
from json_database import JsonStorageXDG
from nltk.corpus import brown

db = JsonStorageXDG("nltk_brown_brill_tagger", subfolder="ModelZoo/nltk")

MODEL_META = {
    "corpus": "brown",
    "model_id": "nltk_brown_brill_tagger",
    "tagset": "brown",
    "corpus_homepage": "http://www.hit.uib.no/icame/brown/bcm.html",
    "lang": "en",
    "algo": "nltk.brill.fntbl37",
    "backoff_taggers": ["AffixTagger", "UnigramTagger", "BigramTagger",
                        "TrigramTagger"],
    "required_packages": ["nltk"]
}
db.update(MODEL_META)
db.store()
model_path = db.path.replace(".json", ".pkl")

# initializing training and testing set
nltk.download('brown')

corpus = [_ for _ in brown.tagged_sents()]  # 57340
random.shuffle(corpus)
cuttof = int(len(corpus) * 0.9)
train_data = corpus[:cuttof]
test_data = corpus[cuttof:]

ngram_tagger = joblib.load(model_path.replace("brill", "ngram"))

tagger = nltk.BrillTaggerTrainer(ngram_tagger, nltk.brill.fntbl37())
tagger = tagger.train(train_data, max_rules=100)

a = tagger.evaluate(test_data)
print("Accuracy of Brill tagger : ", a)  # 0.9353010205150772
db["accuracy"] = a
db.store()
joblib.dump(tagger, model_path)
