from random import shuffle

import joblib
import nltk
from json_database import JsonStorageXDG
from nltk.tag import PerceptronTagger

db = JsonStorageXDG("nltk_cess_cat_perceptron_tagger", subfolder="ModelZoo/nltk")
MODEL_META = {
    "corpus": "cess_cat",
    "corpus_homepage": "https://web.archive.org/web/20121023154634/http://clic.ub.edu/cessece/",
    "lang": "ca",
    "model_id": "nltk_cess_cat_perceptron_tagger",
    "tagset": "EAGLES",
    "tagset_homepage": "http://www.ilc.cnr.it/EAGLES96/annotate/annotate.html",
    "algo": "Perceptron",
    "required_packages": ["nltk"]
}
db.update(MODEL_META)
db.store()
model_path = db.path.replace(".json", ".pkl")

# EAGLES
# http://www.ilc.cnr.it/EAGLES96/annotate/annotate.html
nltk.download('cess_cat')

corpus = [sent for sent in nltk.corpus.cess_cat.tagged_sents()]
shuffle(corpus)
cutoff = int(len(corpus) * 0.9)
train_data = corpus[:cutoff]
test_data = corpus[cutoff:]

tagger = PerceptronTagger(load=False)
tagger.train(train_data)

a = tagger.evaluate(test_data)

print("Accuracy of tagger : ", a)  # 0.9384222700805616
db["accuracy"] = a

joblib.dump(tagger, model_path)
