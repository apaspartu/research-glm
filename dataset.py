import os
import random
from misc import Corpus, Text, LearningMode, Dataset, extract_words

CORPUS_DIR = './corpus/tokenized'

PHRASE_SIZE = 4  # number of words in phrase


def load_corpus():
    if not os.path.exists(CORPUS_DIR):
        raise FileNotFoundError('Corpus directory not exists')

    texts = []

    total_number_of_sentences = 0
    total_number_of_characters = 0

    vocabulary = set()

    for file_name in sorted(os.listdir(CORPUS_DIR)):
        with open(os.path.join(CORPUS_DIR, file_name)) as file:
            content = file.read()

        vocabulary = vocabulary.union(content).difference('\n')

        sentences = list(map(str.strip, content.split('\n')))

        number_of_sentences = len(sentences)
        number_of_characters = sum(len(sentence) for sentence in sentences)

        total_number_of_sentences += number_of_sentences
        total_number_of_characters += number_of_characters

        texts.append(Text(sentences, number_of_sentences, number_of_characters))

    return Corpus(texts, list(sorted(vocabulary)), total_number_of_sentences, total_number_of_characters)


def build_sentence_samples(corpus: Corpus, partition: float):
    def training_set_generator():
        for text in corpus.texts:
            training_set_size = int(len(text.sentences) * partition)
            sentences = text.sentences[:training_set_size]

            for sentence_index in range(len(sentences) - 1):
                yield sentences[sentence_index], sentences[sentence_index + 1]

    def test_set_generator():
        for text in corpus.texts:
            training_set_size = int(len(text.sentences) * partition)
            sentences = text.sentences[training_set_size:]

            for sentence_index in range(len(sentences) - 1):
                yield sentences[sentence_index], sentences[sentence_index + 1]

    return training_set_generator, test_set_generator


def build_word_samples(corpus: Corpus, partition: float, shuffle=True):
    def training_set_generator():
        for text in corpus.texts:
            training_set_size = int(len(text.sentences) * partition)
            sentences = text.sentences[:training_set_size]

            if shuffle:
                random.shuffle(sentences)

            for sentence in sentences:
                words = extract_words(sentence)
                yield from zip(words, words)

    def test_set_generator():
        for text in corpus.texts:
            training_set_size = int(len(text.sentences) * partition)
            sentences = text.sentences[training_set_size:]

            if shuffle:
                random.shuffle(sentences)

            for sentence in sentences:
                yield from extract_words(sentence)

    return training_set_generator, test_set_generator


def build_phrase_samples(corpus: Corpus, partition: float, shuffle=True):
    def training_set_generator():
        for text in corpus.texts:
            training_set_size = int(len(text.sentences) * partition)
            sentences = text.sentences[:training_set_size]

            if shuffle:
                random.shuffle(sentences)

            for sentence in sentences:
                words = extract_words(sentence)

                for target_word in range(PHRASE_SIZE - 1, len(words) - PHRASE_SIZE):
                    source_phrase = words[target_word - (PHRASE_SIZE - 1):target_word + 1]
                    target_phrase = words[target_word + 1:target_word + PHRASE_SIZE + 1]

                    yield ' '.join(source_phrase), ' '.join(target_phrase)

    def test_set_generator():
        for text in corpus.texts:
            training_set_size = int(len(text.sentences) * partition)
            sentences = text.sentences[training_set_size:]

            if shuffle:
                random.shuffle(sentences)

            for sentence in sentences:
                words = extract_words(sentence)

                for target_word in range(PHRASE_SIZE - 1, len(words) - PHRASE_SIZE):
                    source_phrase = words[target_word - (PHRASE_SIZE - 1):target_word + 1]
                    target_phrase = words[target_word + 1:target_word + PHRASE_SIZE + 1]

                    yield ' '.join(source_phrase), ' '.join(target_phrase)

    return training_set_generator, test_set_generator


def build_dataset(partition=0.9, mode=LearningMode.SENTENCE, shuffle=True):
    if partition <= 0 or partition >= 1:
        raise ValueError('Partition must be in (0, 1)')

    corpus = load_corpus()

    if mode == LearningMode.SENTENCE:
        training_set, test_set = build_sentence_samples(corpus, partition)
    elif mode == LearningMode.WORD:
        training_set, test_set = build_word_samples(corpus, partition, shuffle)
    elif mode == LearningMode.PHRASE:
        training_set, test_set = build_phrase_samples(corpus, partition, shuffle)
    else:
        raise Exception('Invalid learning mode')

    return Dataset(corpus, training_set, test_set)
