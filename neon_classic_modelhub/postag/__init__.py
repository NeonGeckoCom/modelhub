

class AbstractPostagger:
    def tag(self, sentence):
        return [(w, "UNK") for w in sentence.split()]

