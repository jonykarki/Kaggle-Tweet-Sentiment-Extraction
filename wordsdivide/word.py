class Word:
    def __init__(self, word, sentiment):
        self.word = word.lower()
        self.sentiment = sentiment

    def __lt__(self, other):
        return self.word < other.word

    def __repr__(self):
        return self.word
