import random

import joblib
import nltk
from json_database import JsonStorageXDG
from nltk.corpus import brown
from nltk.tag.sequential import ClassifierBasedPOSTagger

db = JsonStorageXDG("nltk_brown_udep_nb_tagger", subfolder="ModelZoo/nltk")
MODEL_META = {
    "corpus": "brown",
    "lang": "en",
    "corpus_homepage": "http://www.hit.uib.no/icame/brown/bcm.html",
    "model_id": "nltk_brown_udep_nb_tagger",
    "tagset": "Universal Dependencies",
    "algo": "NaiveBayes",
    "required_packages": ["nltk"]
}
db.update(MODEL_META)
db.store()
model_path = db.path.replace(".json", ".pkl")

# initializing training and testing set
nltk.download('brown')
nltk.download('universal_tagset')

corpus = list(brown.tagged_sents(tagset='universal'))  # 3914
random.shuffle(corpus)
train_data = corpus[:3000]
test_data = corpus[3000:]

tagger = ClassifierBasedPOSTagger(train=train_data)

a = tagger.evaluate(test_data)

print("Accuracy: ", a)
db["accuracy"] = a
db.store()
joblib.dump(tagger, model_path)
