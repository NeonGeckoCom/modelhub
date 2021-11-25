from random import shuffle
from nltk.tag import PerceptronTagger
import joblib
import nltk
from json_database import JsonStorageXDG

db = JsonStorageXDG("nltk_floresta_perceptron_tagger", subfolder="ModelZoo/nltk")

MODEL_META = {
    "corpus": "floresta",
    "corpus_homepage": "http://www.linguateca.pt/Floresta",
    "model_id": "nltk_floresta_perceptron_tagger",
    "tagset": "VISL (Portuguese)",
    "tagset_homepage": "https://visl.sdu.dk/visl/pt/symbolset-floresta.html",
    "lang": "pt",
    "algo": "Perceptron",
    "required_packages": ["nltk"]
}
db.update(MODEL_META)
db.store()
model_path = db.path.replace(".json", ".pkl")

nltk.download('floresta')


def clean_tag(t):
    if "+" in t: t = t.split("+")[1]
    if "|" in t: t = t.split("|")[1]
    if "#" in t: t = t.split("#")[0]
    t = t.lower()
    return t


floresta = [[(w, clean_tag(t)) for (w, t) in sent]
            for sent in nltk.corpus.floresta.tagged_sents()]
shuffle(floresta)

cutoff = int(len(floresta) * 0.9)
train_data = floresta[:cutoff]
test_data = floresta[cutoff:]

tagger = PerceptronTagger(load=False)
tagger.train(train_data)

a = tagger.evaluate(test_data)

print("Accuracy of perceptron tagger : ", a)  # 0.9272517853029256
db["accuracy"] = a
db.store()
joblib.dump(tagger, model_path)
