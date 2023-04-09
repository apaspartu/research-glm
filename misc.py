import re
from enum import Enum
from typing import Generator, Callable, Any, Iterable


class Text:
    def __init__(self, sentences: list[str], sentences_count: int, characters_count: int):
        self.sentences = sentences
        self.sentences_count = sentences_count
        self.characters_count = characters_count


class Metadata:
    def __init__(self):
        self.vocabulary = None
        self.total_sentences_count = None
        self.total_characters_count = None


class Corpus:
    def __init__(self, texts: list[Text], vocabulary: Iterable[str],
                 total_sentences_count: int, total_characters_count: int):
        self.texts = texts
        self.vocabulary = vocabulary
        self.total_sentences_count = total_sentences_count
        self.total_characters_count = total_characters_count


class Dataset:
    def __init__(self, corpus: Corpus,
                 training_set: Callable[[], Generator[tuple[str, str], Any, None]],
                 test_set: Callable[[], Generator[tuple[str, str], Any, None]]):
        self.training_set = training_set
        self.test_set = test_set
        self.corpus = corpus


class LearningMode(Enum):
    WORD = 'word'
    PHRASE = 'phrase'
    SENTENCE = 'sentence'


def extract_words(sentence: str) -> list[str]:
    word_pattern = r'\w+'
    return re.findall(word_pattern, sentence)
