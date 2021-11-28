from nltk import pos_tag, word_tokenize

from neon_modelhub import load_model
from neon_modelhub.postag import AbstractPostagger


class NltkPostag(AbstractPostagger):
    def __init__(self, tagger = None):
        self.tagger = tagger

    def tag(self, sentence):
        if isinstance(sentence, str):
            toks = word_tokenize(sentence)
        else:
            toks = sentence
        if self.tagger:
            return self.tagger.tag(toks)
        return pos_tag(toks)


def get_default_postagger(lang="en"):
    tagger = None
    if lang == "en":
        model_id = ""  # use nltk defaults (?)
    if lang == "pt":
        model_id = "nltk_macmorpho_udep_perceptron_tagger"
        tagger = load_model(model_id)
    if lang == "es":
        model_id = "nltk_cess_esp_udep_perceptron_tagger"
        tagger = load_model(model_id)
    if lang == "ca":
        model_id = "nltk_cess_cat_udep_perceptron_tagger"
        tagger = load_model(model_id)
    return NltkPostag(tagger)


