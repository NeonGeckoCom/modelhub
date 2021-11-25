from nltk.chunk import ChunkParserI
from nltk.chunk import conlltags2tree

from neon_modelhub.chunkers import AbstractChunker
from neon_modelhub.features.nltk_feats import NltkFeatures
from neon_modelhub.postag.nltk_postag import get_default_postagger


class BaseNltkChunker(AbstractChunker, ChunkParserI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.tagger:
            self.tagger = get_default_postagger(self.lang)


class NamedEntityChunker(BaseNltkChunker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_detector = NltkFeatures.extract_iob_features

    def parse(self, sentence):
        chunks = self.tag(sentence)

        # Transform the result from [((w1, t1), iob1), ...]
        # to the preferred list of triplets format [(w1, t1, iob1), ...]
        iob_triplets = [(w, t, c) for ((w, t), c) in chunks]

        # Transform the list of triplets to nltk.Tree format
        return conlltags2tree(iob_triplets)


class PostagChunkParser(BaseNltkChunker):
    def parse(self, sentence):
        if isinstance(sentence, str):
            sentence = self.tagger.tag(sentence)
        pos_tags = [pos for word, pos in sentence]

        # Get the Chunk tags
        tagged_pos_tags = self.chunker.tag(pos_tags)

        # Assemble the (word, pos, chunk) triplets
        iob_triplets = [(word, pos_tag, chunk_tag)
                        for ((word, pos_tag), (pos_tag, chunk_tag)) in
                        zip(sentence, tagged_pos_tags)]

        # Transform the list of triplets to nltk.Tree format
        return conlltags2tree(iob_triplets)


class ClassifierChunkParser(BaseNltkChunker):

    def parse(self, sentence):
        if isinstance(sentence, str):
            sentence = self.tagger.tag(sentence)

        chunks = self.chunker.tag(sentence)

        # Transform the result from [((w1, t1), iob1), ...]
        # to the preferred list of triplets format [(w1, t1, iob1), ...]
        iob_triplets = [(w, t, c) for ((w, t), c) in chunks]

        # Transform the list of triplets to nltk.Tree format
        return conlltags2tree(iob_triplets)
