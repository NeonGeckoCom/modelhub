from random import shuffle
from json_database import JsonStorageXDG
import nltk
import joblib


db = JsonStorageXDG("nltk_cess_esp_brill_tagger", subfolder="ModelZoo/nltk")
MODEL_META = {
    "corpus": "cess_esp",
    "corpus_homepage": "https://web.archive.org/web/20121023154634/http://clic.ub.edu/cessece/",
    "lang": "es",
    "model_id": "nltk_cess_esp_brill_tagger",
    "tagset": "EAGLES",
    "tagset_homepage": "http://www.ilc.cnr.it/EAGLES96/annotate/annotate.html",
    "algo": "nltk.brill.fntbl37",
    "backoff_taggers": ["AffixTagger", "UnigramTagger",
                        "BigramTagger", "TrigramTagger"],
    "required_packages": ["nltk"]
}
db.update(MODEL_META)
db.store()
model_path = db.path.replace(".json", ".pkl")

# EAGLES
# http://www.ilc.cnr.it/EAGLES96/annotate/annotate.html
nltk.download('cess_esp')

corpus = [sent for sent in nltk.corpus.cess_cat.tagged_sents()]
shuffle(corpus)
cutoff = int(len(corpus) * 0.9)
train_data = corpus[:cutoff]
test_data = corpus[cutoff:]

ngram_tagger = joblib.load(model_path.replace("brill", "ngram"))

tagger = nltk.BrillTaggerTrainer(ngram_tagger, nltk.brill.fntbl37())
tagger = tagger.train(train_data)

a = tagger.evaluate(test_data)

print("Accuracy of Brill tagger : ", a)  # 0.9384222700805616
db["accuracy"] = a

joblib.dump(tagger, model_path)
