from os.path import join, dirname
import random

import joblib
from json_database import JsonStorageXDG
from nltk.corpus import treebank
from nltk.tag import DefaultTagger
from nltk.tag import UnigramTagger, BigramTagger, TrigramTagger
from nltk.tag import DefaultTagger
from nltk.tag import RegexpTagger
import joblib
import nltk
from nltk import AffixTagger
from nltk.corpus import treebank
from nltk.tag import DefaultTagger
from nltk.tag import RegexpTagger
from nltk.tag import UnigramTagger, BigramTagger, TrigramTagger
from nltk.tag import brill, brill_trainer

db = JsonStorageXDG("nltk_treebank_brill_tagger", subfolder="ModelZoo/nltk")
MODEL_META = {
    "corpus": "treebank",
    "corpus_homepage": "http://www.hit.uib.no/icame/brown/bcm.html",
    "model_id": "nltk_treebank_brill_tagger",
    "tagset": "Penn Treebank",
    "lang": "en",
    "algo": "BrillTaggerTrainer",
    "backoff_taggers": ["DefaultTagger", "AffixTagger", "RegexpTagger",
                        "UnigramTagger", "BigramTagger", "TrigramTagger"],
    "required_packages": ["nltk"]
}
db.update(MODEL_META)
db.store()
model_path = db.path.replace(".json", ".pkl")

corpus = list(treebank.tagged_sents())  # 3914
random.shuffle(corpus)
train_data = corpus[:3000]
test_data = corpus[3000:]


ngram_tagger = joblib.load(model_path.replace("brill", "ngram"))


def train_brill_tagger(initial_tagger, train_sents, **kwargs):
    templates = [
        brill.Template(brill.Pos([-1])),
        brill.Template(brill.Pos([1])),
        brill.Template(brill.Pos([-2])),
        brill.Template(brill.Pos([2])),
        brill.Template(brill.Pos([-2, -1])),
        brill.Template(brill.Pos([1, 2])),
        brill.Template(brill.Pos([-3, -2, -1])),
        brill.Template(brill.Pos([1, 2, 3])),
        brill.Template(brill.Pos([-1]), brill.Pos([1])),
        brill.Template(brill.Word([-1])),
        brill.Template(brill.Word([1])),
        brill.Template(brill.Word([-2])),
        brill.Template(brill.Word([2])),
        brill.Template(brill.Word([-2, -1])),
        brill.Template(brill.Word([1, 2])),
        brill.Template(brill.Word([-3, -2, -1])),
        brill.Template(brill.Word([1, 2, 3])),
        brill.Template(brill.Word([-1]), brill.Word([1])),
    ]

    trainer = brill_trainer.BrillTaggerTrainer(initial_tagger, templates,
                                               deterministic=True)
    return trainer.train(train_sents, **kwargs)


tagger = train_brill_tagger(ngram_tagger, train_data)

a = tagger.evaluate(test_data)

print("Accuracy of Brill tagger : ", a)  # 0.9083099503561407
db["accuracy"] = a
db.store()
joblib.dump(tagger, model_path)
