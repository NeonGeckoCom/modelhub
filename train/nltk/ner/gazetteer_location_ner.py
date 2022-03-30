import joblib
from json_database import JsonStorageXDG
from nltk.chunk import conlltags2tree
from nltk.corpus import gazetteers
import nltk
from neon_classic_modelhub.chunkers.nltk_chunkers import NamedEntityChunker

db = JsonStorageXDG("nltk_locations_gazetteer_ner", subfolder="ModelZoo/nltk")
MODEL_META = {
    "model_id": "nltk_locations_gazetteer_ner",
    "algo": "gazetteer",
    "required_packages": ["nltk"]
}
db.update(MODEL_META)
db.store()
model_path = db.path.replace(".json", ".pkl")


class GazetteerLocationChunker(NamedEntityChunker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nltk.download('gazetteers')
        self.locations = set(gazetteers.words())
        longest_word = max(self.locations, key=len)
        self.lookahead = len(longest_word.split(" "))

    def iob_locations(self, sentence):
        if isinstance(sentence, str):
            sentence = self.tagger.tag(sentence)

        found_locs = []
        for idx, (word, tag) in enumerate(sentence):
            nexttags = sentence[idx + 1:idx + self.lookahead]
            for i in range(len(nexttags)):
                loc_tags = nexttags[:i]
                loc_str = " ".join([word] + [n[0] for n in loc_tags])

                if loc_str in self.locations:
                    found_locs.append(idx)
                    yield word, tag, 'B-LOCATION'
                    for i, (w, t) in enumerate(loc_tags):
                        found_locs.append(i + idx + 1)
                        yield w, t, 'I-LOCATION'

                    break
            else:
                if word in self.locations:
                    found_locs.append(idx)
                    yield word, tag, 'B-LOCATION'

            if idx not in found_locs:
                yield word, tag, 'O'

    def tag(self, sentence):
        return list(self.iob_locations(sentence))

    def parse(self, sentence):
        iobs = self.iob_locations(sentence)
        return conlltags2tree(iobs)


chunker = GazetteerLocationChunker()

joblib.dump(chunker, model_path)

