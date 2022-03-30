import joblib
from json_database import JsonStorageXDG
from nltk.tag import ClassifierBasedTagger

from neon_classic_modelhub.chunkers.nltk_chunkers import NamedEntityChunker, \
    conlltags2tree
from neon_classic_modelhub.features.nltk_feats import NltkFeatures

db = JsonStorageXDG("nltk_CONLL2003_nb_NER", subfolder="ModelZoo/nltk")
MODEL_META = {
    "corpus": "CONLL2003",
    "lang": "en",
    "corpus_homepage": "https://github.com/davidsbatista/NER-datasets/tree/master/CONLL2003",
    "model_id": "nltk_CONLL2003_nb_NER",
    "tagset": "conll_iob",
    "algo": "NaiveBayes",
    "entitíes": ['ORG', 'LOC', 'MISC', 'PER'],
    "required_packages": ["nltk", "neon_classic_modelhub"]
}
db.update(MODEL_META)
db.store()
model_path = db.path.replace(".json", ".pkl")


# corpus handling
def read_connll2003(filename):
    with open(filename, 'r') as file_handle:
        file_content = file_handle.read()
        annotated_sentences = file_content.split('\n\n')
        for annotated_sentence in annotated_sentences:
            annotated_tokens = [seq for seq in
                                annotated_sentence.split('\n') if
                                seq]

            standard_form_tokens = []
            for idx, annotated_token in enumerate(
                    annotated_tokens):
                annotations = annotated_token.split(' ')
                word, tag, ner = annotations[0], annotations[1], \
                                 annotations[3]
                standard_form_tokens.append((word, tag, ner))

            # Make it NLTK Classifier compatible - [(w1, t1, iob1), ...] to [((w1, t1), iob1), ...]
            # Because the classfier expects a tuple as input, first item input, second the class
            yield [((w, t), iob) for w, t, iob in
                   standard_form_tokens]


corpus_root = "/home/user/PycharmProjects/nlp_workspace/biblioteca/corpora/NER-datasets/CONLL2003/train.txt"
reader = read_connll2003(corpus_root)
training_samples = list(reader)

corpus_root = "/home/user/PycharmProjects/nlp_workspace/biblioteca/corpora/NER-datasets/CONLL2003/test.txt"
reader = read_connll2003(corpus_root)
test_samples = list(reader)


def train():
    # training
    tagger = ClassifierBasedTagger(
        train=training_samples,
        feature_detector=NltkFeatures.extract_iob_features)

    joblib.dump(tagger, model_path)


def accuracy_test():
    chunker = NamedEntityChunker(model_id=MODEL_META["model_id"])

    # accuracy test
    score = chunker.evaluate(
        [conlltags2tree([(w, t, iob) for (w, t), iob in iobs])
         for iobs in test_samples])
    a = score.accuracy()
    print("Accuracy : ", a)
    db["accuracy"] = a
    db.store()


#train()
accuracy_test()  # 0.8743839197702824


chunker = NamedEntityChunker(model_id=MODEL_META["model_id"])
r = chunker.parse("London is a great city")
print(r)
