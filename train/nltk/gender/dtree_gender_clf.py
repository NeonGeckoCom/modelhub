import random

import joblib
import nltk
from json_database import JsonStorageXDG
from nltk import DecisionTreeClassifier
from nltk.classify import accuracy
from nltk.corpus import names

from neon_modelhub.features.nltk_feats import NltkFeatures

db = JsonStorageXDG("nltk_dtree_gender_clf", subfolder="ModelZoo/nltk")
MODEL_META = {
    "corpus": "nltk names",
    "lang": "en",
    "corpus_homepage": "",
    "model_id": "nltk_dtree_gender_clf",
    "algo": "ClassifierBasedTagger",
    "required_packages": ["nltk", "JarbasModelZoo"]
}
db.update(MODEL_META)
db.store()
model_path = db.path.replace(".json", ".pkl")

nltk.download('names')

corpus = ([(name, 'male') for name in names.words('male.txt')] +
          [(name, 'female') for name in names.words('female.txt')])
corpus = [(NltkFeatures.extract_single_word_features(n), gender) for (n, gender) in corpus]

random.shuffle(corpus)

cutoff = int(len(corpus) * 0.8)
train_data = corpus[:cutoff]
test_data = corpus[cutoff:]

classifier = DecisionTreeClassifier.train(train_data)
print(classifier.pseudocode(depth=4))

a = accuracy(classifier, test_data)
print(a)  # 0.8489616110761485
db["accuracy"] = a
db.store()
joblib.dump(classifier, model_path)
