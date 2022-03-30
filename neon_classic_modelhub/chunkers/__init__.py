from neon_classic_modelhub import load_model


class AbstractChunker:
    def __init__(self, model_id=None, tagger_id=None, lang="en"):
        self.lang = lang
        if tagger_id:
            self.tagger = load_model(tagger_id)
        else:
            self.tagger = None
        if model_id:
            self.chunker = load_model(model_id)
        else:
            self.chunker = None

    def parse(self, sentence):
        if isinstance(sentence, str):
            sentence = self.tagger.tag(sentence)
        return self.chunker.parse(sentence)

    def tag(self, sentence):
        if isinstance(sentence, str):
            sentence = self.tagger.tag(sentence)
        return self.chunker.tag(sentence)
