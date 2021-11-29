from random import shuffle

import joblib
from json_database import JsonStorageXDG
from nltk.tag import PerceptronTagger

import biblioteca

biblioteca.download("cess_esp_udep")
cess = biblioteca.load_corpus("cess_esp_udep")

db = JsonStorageXDG("nltk_cess_esp_udep_perceptron_tagger", subfolder="ModelZoo/nltk")
MODEL_META = {
    "corpus": "cess_esp_udep",
    "corpus_homepage": "https://web.archive.org/web/20121023154634/http://clic.ub.edu/cessece/",
    "lang": "es",
    "model_id": "nltk_cess_esp_udep_perceptron_tagger",
    "tagset": "Universal Dependencies",
    "algo": "Perceptron",
    "required_packages": ["nltk"]
}
db.update(MODEL_META)
db.store()
model_path = db.path.replace(".json", ".pkl")

corpus = list(cess.tagged_sentences())
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
