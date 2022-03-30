import random

import joblib
import nltk
from json_database import JsonStorageXDG
from nltk.classify import MaxentClassifier
from nltk.classify import accuracy
from nltk.corpus import names

from neon_classic_modelhub.features.nltk_feats import NltkFeatures

db = JsonStorageXDG("nltk_gender_maxent_clf", subfolder="ModelZoo/nltk")
MODEL_META = {
    "corpus": "nltk names",
    "lang": "en",
    "corpus_homepage": "",
    "model_id": "nltk_gender_maxent_clf",
    "algo": "MaxentClassifier",
    "required_packages": ["nltk", "neon_classic_modelhub"]
}
db.update(MODEL_META)
db.store()
model_path = db.path.replace(".json", ".pkl")

nltk.download('names')

corpus = ([(name, 'male') for name in names.words('male.txt')] +
          [(name, 'female') for name in names.words('female.txt')])
corpus = [(NltkFeatures.extract_single_word_features(n), gender)
          for (n, gender) in corpus]

random.shuffle(corpus)

cutoff = int(len(corpus) * 0.8)
train_data = corpus[:cutoff]
test_data = corpus[cutoff:]

classifier = MaxentClassifier.train(train_data)
a = accuracy(classifier, test_data)
print(a)  # 0.8489616110761485
db["accuracy"] = a
db.store()
joblib.dump(classifier, model_path)
