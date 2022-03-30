import re

import nltk
from nltk.stem.snowball import SnowballStemmer


class NltkFeatures:
    @staticmethod
    def get_word_shape(word):
        word_shape = 'other'
        if re.match('[0-9]+(\.[0-9]*)?|[0-9]*\.[0-9]+$', word):
            word_shape = 'number'
        elif re.match('\W+$', word):
            word_shape = 'punct'
        elif re.match('[A-Z][a-z]+$', word):
            word_shape = 'capitalized'
        elif re.match('[A-Z]+$', word):
            word_shape = 'uppercase'
        elif re.match('[a-z]+$', word):
            word_shape = 'lowercase'
        elif re.match('[A-Z][a-z]+[A-Z][a-z]+[A-Za-z]*$', word):
            word_shape = 'camelcase'
        elif re.match('[A-Za-z]+$', word):
            word_shape = 'mixedcase'
        elif re.match('__.+__$', word):
            word_shape = 'wildcard'
        elif re.match('[A-Za-z0-9]+\.$', word):
            word_shape = 'ending-dot'
        elif re.match('[A-Za-z0-9]+\.[A-Za-z0-9\.]+\.$', word):
            word_shape = 'abbreviation'
        elif re.match('[A-Za-z0-9]+\-[A-Za-z0-9\-]+.*$', word):
            word_shape = 'contains-hyphen'

        return word_shape

    @staticmethod
    def extract_coref_features(tokens, index, history, stemmer=None, memory=3, look_ahead=10):
        """
        `tokens`  = a POS-tagged sentence [(w1, t1), ...]
        `index`   = the index of the token we want to extract features for
        `history` = the previous predicted IOB tags
        """
        feat_dict = {}

        male_words = ["boy", "man", "men", "male"] + \
                     ["father", "grandfather", "brother", "cousin", "uncle", "son", "dad"]
        female_words = ["girl", "woman", "women", "female"] + \
                       ["mother", "grandmother", "sister", "cousin", "aunt", "daughter", "mom"]
        male_words += [m + "s" for m in male_words]
        female_words += [f + "s" for f in female_words]

        male_prons = ["he", "him", "his"]
        singular_prons = ["i", "my"]
        female_prons = ["she", "her", "hers"]
        nonhuman_prons = ["it", "that", "this", "these", "which", "them"]
        neutral_prons = ["you", "yours", "your", "they", "them", "their", "theirs", "who", "whose", "whom"]
        plural_prons = ["they", "them", "their", "theirs"]

        plural_suffix = "s"
        valid_pronouns = male_prons + female_prons + neutral_prons + nonhuman_prons + plural_prons

        def get_current_word_feats():
            # is this word a pronoun
            word, tag = tokens[index]
            word = word.lower()

            prev_tag = prev_prev_tag = "."
            next_tag = next_next_tag = "."
            if index >= 1:
                _, prev_tag = tokens[index - 1]
            if index >= 2:
                _, prev_prev_tag = tokens[index - 2]
            if index + 1 < len(tokens):
                _, next_tag = tokens[index + 1]
            if index + 2 < len(tokens):
                _, next_next_tag = tokens[index + 2]

            """
            # helps in detecting multi word names
            if tag == "NOUN" and (prev_tag == "NOUN" or next_tag == "NOUN"):
                feat_dict["multi_word"] = True
            # helps in detecting plurals
            elif (tag == "NOUN" and prev_tag == "CONJ" and prev_prev_tag == "NOUN") or \
                    (tag == "NOUN" and next_tag == "CONJ" and next_next_tag == "NOUN") or \
                    (tag == "CONJ" and prev_tag == "NOUN" and next_tag == "NOUN"):
                #feat_dict["is_conjunction"] = True
                feat_dict["multi_word"] = True
            """

            # help disambiguating pronouns
            # this is hacky and wordlist based...
            if word in female_words:
                feat_dict["is_female"] = True
            if word in male_words:
                feat_dict["is_male"] = True
            if word.endswith(plural_suffix):
                feat_dict["plural_suffix"] = True

            # if tag == "PRON":
            #    feat_dict["is_pronoun"] = True
            # elif tag == "NOUN":
            #    feat_dict["is_noun"] = True
            #if word in valid_pronouns:
            #    feat_dict["is_pronoun_word"] = True
            if word in singular_prons:
                feat_dict["is_singular_pronoun_word"] = True
            if word in male_prons:
                feat_dict["is_male_pronoun_word"] = True
            if word in female_prons:
                feat_dict["is_female_pronoun_word"] = True
            if word in neutral_prons:
                feat_dict["is_neutral_pronoun_word"] = True
            if word in plural_prons:
                feat_dict["is_plural_pronoun_word"] = True
            if word in nonhuman_prons:
                feat_dict["is_inanimate_pronoun_word"] = True

        def get_prev_noun_feats():
            # are there nouns before this word
            nouns = [(w, idx) for idx, (w, t) in enumerate(tokens[:index]) if t == "NOUN"]
            nouns.reverse()
            for idx, (nword, i) in enumerate(nouns):
                dst = index - i
                if dst >= look_ahead:
                    break
                feat_dict[f"prev_noun_{idx}"] = nword
                feat_dict[f"prev_noun_{idx}_dist"] = dst
                feat_dict[f"prev_noun_{idx}_plural_suffix"] = nword.endswith(plural_suffix)
                feat_dict["prev_noun_capitalized"] = nword[0].isupper()

        def get_next_pronoun_feats():
            # closest pronouns after this word
            female = False
            male = False
            neutral = False
            plural = False
            inanimate = False
            for i in range(index + 1, len(tokens)):
                if i >= len(tokens) or i >= look_ahead:
                    break

                nword, ntag = tokens[i]
                nword = nword.lower()
                if nword in female_prons and not female:
                    # feat_dict["next_female_pronoun"] = True
                    feat_dict["next_female_pronoun_dist"] = i
                    female = True
                if nword in male_prons and not male:
                    # feat_dict["next_male_pronoun"] = True
                    feat_dict["next_male_pronoun_dist"] = i
                    male = True
                if nword in neutral_prons and not neutral:
                    # feat_dict["next_neutral_pronoun"] = True
                    feat_dict["next_neutral_pronoun_dist"] = i
                    neutral = True
                if nword in plural_prons and not plural:
                    # feat_dict["next_plural_pronoun"] = True
                    feat_dict["next_plural_pronoun_dist"] = i
                    plural = True
                if nword in nonhuman_prons and not inanimate:
                    # feat_dict["next_inanimate_pronoun"] = True
                    feat_dict["next_inanimate_pronoun_dist"] = i
                    inanimate = True

        get_current_word_feats()
        # get_prev_noun_feats()
        get_next_pronoun_feats()

        iobdict = NltkFeatures.extract_iob_features(tokens, index, history,
                                                    stemmer=stemmer, memory=memory)
        return {**iobdict, **feat_dict}

    @staticmethod
    def extract_iob_features(tokens, index, history, stemmer=None, memory=3):
        """
        `tokens`  = a POS-tagged sentence [(w1, t1), ...]
        `index`   = the index of the token we want to extract features for
        `history` = the previous predicted IOB tags
        """
        feat_dict = NltkFeatures.extract_postag_features(tokens, index, stemmer=stemmer,
                                                         memory=memory)
        # Pad the sequence with placeholders
        tokens = ['O'] * memory + history

        index += memory

        # look back N predictions
        for i in range(1, memory + 1):
            k = "prev-" * i
            previob = tokens[index - i]
            # update with IOB features
            feat_dict[k + "iob"] = previob

        return feat_dict

    @staticmethod
    def extract_postag_features(tokens, index, stemmer=None, memory=3):
        """
        `tokens`  = a POS-tagged sentence [(w1, t1), ...]
        `index`   = the index of the token we want to extract features for
        """
        original_toks = list(tokens)
        # word features
        feat_dict = NltkFeatures.extract_word_features([t[0] for t in tokens],
                                                       index, stemmer, memory=memory)

        # Pad the sequence with placeholders
        tokens = []
        for i in range(1, memory + 1):
            tokens.append((f'__START{i}__', f'__START{i}__'))
        tokens = list(reversed(tokens)) + original_toks
        for i in range(1, memory + 1):
            tokens.append((f'__END{i}__', f'__END{i}__'))

        # shift the index to accommodate the padding
        index += memory

        word, pos = tokens[index]

        # update with postag features
        feat_dict["pos"] = pos

        # look ahead N words
        for i in range(1, memory + 1):
            k = "next-" * i
            nextword, nextpos = tokens[index + i]
            feat_dict[k + "pos"] = nextpos

        # look back N words
        for i in range(1, memory + 1):
            k = "prev-" * i
            prevword, prevpos = tokens[index - i]
            feat_dict[k + "pos"] = prevpos

        return feat_dict

    @staticmethod
    def extract_word_features(tokens, index=0, stemmer=None, memory=3):
        """
        `tokens`  = a tokenized sentence [w1, w2, ...]
        `index`   = the index of the token we want to extract features for
        """
        if isinstance(tokens, str):
            tokens = [tokens]
        original_toks = list(tokens)
        # init the stemmer
        stemmer = stemmer or SnowballStemmer('english')

        # Pad the sequence with placeholders
        tokens = []
        for i in range(1, memory + 1):
            tokens.append(f'__START{i}__')
        tokens = list(reversed(tokens)) + original_toks
        for i in range(1, memory + 1):
            tokens.append(f'__END{i}__')

        # shift the index to accommodate the padding
        index += memory

        word = tokens[index]
        feat_dict = NltkFeatures.extract_single_word_features(word)
        feat_dict["word"] = word
        feat_dict["shape"] = NltkFeatures.get_word_shape(word)
        feat_dict["lemma"] = stemmer.stem(word)

        # look ahead N words
        for i in range(1, memory + 1):
            k = "next-" * i
            nextword = tokens[index + i]
            feat_dict[k + "word"] = nextword
            feat_dict[k + "lemma"] = stemmer.stem(nextword)
            feat_dict[k + "shape"] = NltkFeatures.get_word_shape(nextword)

        # look back N words
        for i in range(1, memory + 1):
            k = "prev-" * i
            prevword = tokens[index - i]
            feat_dict[k + "word"] = prevword
            feat_dict[k + "lemma"] = stemmer.stem(prevword)
            feat_dict[k + "shape"] = NltkFeatures.get_word_shape(prevword)

        return feat_dict

    @staticmethod
    def extract_single_word_features(word):
        return {
            'suffix1': word[-1:],
            'suffix2': word[-2:],
            'suffix3': word[-3:],
            'prefix1': word[:1],
            'prefix2': word[:2],
            'prefix3': word[:3]
        }

    @staticmethod
    def extract_rte_features(rtepair):
        extractor = nltk.RTEFeatureExtractor(rtepair)
        features = {}
        features['word_overlap'] = len(extractor.overlap('word'))
        features['word_hyp_extra'] = len(extractor.hyp_extra('word'))
        features['ne_overlap'] = len(extractor.overlap('ne'))
        features['ne_hyp_extra'] = len(extractor.hyp_extra('ne'))
        return features
