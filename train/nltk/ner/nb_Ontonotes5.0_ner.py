import joblib
from json_database import JsonStorageXDG
from nltk.tag import ClassifierBasedTagger

from neon_modelhub.chunkers.nltk_chunkers import NamedEntityChunker, conlltags2tree
from neon_modelhub.features.nltk_feats import NltkFeatures

db = JsonStorageXDG("nltk_onto5_nb_NER", subfolder="ModelZoo/nltk")
MODEL_META = {
    "corpus": "OntoNotes-5.0-NER-BIO",
    "lang": "en",
    "corpus_homepage": "https://github.com/yuchenlin/OntoNotes-5.0-NER-BIO",
    "model_id": "nltk_onto5_nb_NER",
    "tagset": "conll_iob",
    "algo": "NaiveBayes",
    "entities": ['PERSON', 'FAC', 'EVENT', 'GPE', 'TIME',
                 'LAW', 'PRODUCT', 'ORDINAL', 'ORG', 'QUANTITY',
                 'CARDINAL', 'LOC', 'DATE', 'WORK_OF_ART', 'NORP',
                 'LANGUAGE', 'MONEY', 'PERCENT'],
    "required_packages": ["nltk", "neon_modelhub"]
}
db.update(MODEL_META)
db.store()
model_path = db.path.replace(".json", ".pkl")


# corpus handling
def read_ontonotes5(filename):
    if filename.endswith(".ner"):
        with open(filename, 'r') as file_handle:
            file_content = file_handle.read()
            annotated_sentences = file_content.split('\n\n')
            if not annotated_sentences:
                return

            for annotated_sentence in annotated_sentences:
                annotated_tokens = [seq for seq in
                                    annotated_sentence.split('\n') if
                                    seq]
                if not annotated_tokens:
                    continue
                standard_form_tokens = []
                for idx, annotated_token in enumerate(
                        annotated_tokens):
                    annotations = annotated_token.split('\t')
                    word, tag, ner = annotations[0], annotations[1], \
                                     annotations[-1]
                    standard_form_tokens.append((word, tag, ner))

                # Make it NLTK Classifier compatible - [(w1, t1, iob1), ...] to [((w1, t1), iob1), ...]
                # Because the classfier expects a tuple as input, first item input, second the class
                yield [((w, t), iob) for w, t, iob in
                       standard_form_tokens]


corpus_root = "/home/user/PycharmProjects/nlp_workspace/biblioteca/corpora/onto5/onto.train.ner"
reader = read_ontonotes5(corpus_root)
training_samples = list(reader)

corpus_root = "/home/user/PycharmProjects/nlp_workspace/biblioteca/corpora/onto5/onto.test.ner"
reader = read_ontonotes5(corpus_root)
test_samples = list(reader)


def train():
    # training
    tagger = ClassifierBasedTagger(
        train=training_samples,
        feature_detector=NltkFeatures.extract_iob_features)

    joblib.dump(tagger, model_path)


def test():
    chunker = NamedEntityChunker(model_id=MODEL_META["model_id"])

    # accuracy test
    score = chunker.evaluate(
        [conlltags2tree([(w, t, iob) for (w, t), iob in iobs])
         for iobs in test_samples])

    a = score.accuracy()
    print("Accuracy : ", a)  # 0.9353010205150772
    db["accuracy"] = a
    db.store()


train()
test()  # 0.9094512476681258
