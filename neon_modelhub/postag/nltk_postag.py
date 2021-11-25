from nltk import pos_tag, word_tokenize

from neon_modelhub import load_model
from neon_modelhub.postag import AbstractPostagger


class NltkPostag(AbstractPostagger):
    def tag(self, sentence):
        return pos_tag(word_tokenize(sentence))


def get_default_postagger(lang="en"):
    if lang == "en":
        model_id = ""  # use nltk defaults (?)
    if lang == "pt":
        model_id = "nltk_macmorpho_udep_perceptron_tagger"
        return load_model(model_id)
    if lang == "es":
        model_id = "nltk_cess_esp_udep_perceptron_tagger"
        return load_model(model_id)
    if lang == "ca":
        model_id = "nltk_cess_cat_udep_perceptron_tagger"
        return load_model(model_id)
    return NltkPostag()


