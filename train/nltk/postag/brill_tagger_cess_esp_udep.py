from random import shuffle

import joblib
import nltk
from json_database import JsonStorageXDG

import biblioteca
from neon_classic_modelhub import load_model

biblioteca.download("cess_esp_udep")
cess = biblioteca.load_corpus("cess_esp_udep")

db = JsonStorageXDG("nltk_cess_esp_udep_brill_tagger", subfolder="ModelZoo/nltk")
MODEL_META = {
    "corpus": "cess_esp_udep",
    "corpus_homepage": "https://web.archive.org/web/20121023154634/http://clic.ub.edu/cessece/",
    "lang": "es",
    "model_id": "nltk_cess_esp_udep_brill_tagger",
    "tagset": "Universal Dependencies",
    "algo": "nltk.brill.fntbl37",
    "backoff_taggers": ["nltk_cess_esp_udep_ngram_tagger"],
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

ngram_tagger = load_model(model_path.replace("brill", "ngram"))

tagger = nltk.BrillTaggerTrainer(ngram_tagger, nltk.brill.fntbl37())
tagger = tagger.train(train_data)

a = tagger.evaluate(test_data)

print("Accuracy of Brill tagger : ", a)  # 0.9384222700805616
db["accuracy"] = a

joblib.dump(tagger, model_path)
