import joblib
from json_database import JsonStorageXDG
from nltk.chunk import RegexpParser

# PartialRegexChunker
db = JsonStorageXDG("nltk_partial_regex_chunker", subfolder="ModelZoo/nltk")
MODEL_META = {
    "model_id": "nltk_partial_regex_chunker",
    "algo": "regex",
    "required_packages": ["nltk"]
}
db.update(MODEL_META)
db.store()
model_path = db.path.replace(".json", ".pkl")
chunker = RegexpParser(r'''
        NP:
            {<DT>?<NN.*>+}	# chunk optional determiner with nouns
            <JJ>{}<NN.*>	# merge adjective with noun chunk
        PP:
            {<IN>}			# chunk preposition
        VP: 
            {<MD>?<VB.*>}	# chunk optional modal with verb''')
joblib.dump(chunker, model_path)

# ProperNounChunker
db = JsonStorageXDG("nltk_noun_regex_chunker", subfolder="ModelZoo/nltk")
MODEL_META = {
    "model_id": "nltk_nound_regex_chunker",
    "algo": "regex",
    "required_packages": ["nltk"]
}
db.update(MODEL_META)
db.store()
model_path = db.path.replace(".json", ".pkl")
chunker = RegexpParser(r'''NAME:{<NNP>+}''')
joblib.dump(chunker, model_path)
