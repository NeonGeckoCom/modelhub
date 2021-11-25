import random

import joblib
import nltk
from json_database import JsonStorageXDG
from nltk.corpus import brown
from nltk.tag import PerceptronTagger

db = JsonStorageXDG("nltk_brown_perceptron_tagger", subfolder="ModelZoo/nltk")

MODEL_META = {
    "corpus": "brown",
    "model_id": "nltk_brown_perceptron_tagger",
    "tagset": "brown",
    "corpus_homepage": "http://www.hit.uib.no/icame/brown/bcm.html",
    "lang": "en",
    "algo": "Perceptron",
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

tagger = PerceptronTagger(load=False)
tagger.train(train_data)

a = tagger.evaluate(test_data)
print("Accuracy of tagger : ", a)  # 0.9353010205150772
db["accuracy"] = a
db.store()
joblib.dump(tagger, model_path)
