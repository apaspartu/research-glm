import os
import re

ALPHABET = 'АБВГҐДЕЄЖЗИІЇЙКЛМНОПРСТУФХЦЧШЩЬЮЯабвгґдежзиіїйклмнопрстуфхцчшщьюяє' + 'i'  # `i` in the end is english `i`
DIGITS = '0123456789'
PUNCTUATION_MARKS = ['.', ',', '!', '?', '—', ';', ':', '...', '(', ')', '[', ']', '<', '>', '"']
SENTENCE_ENDING_MARKS_ESCAPED = [r'\.', r'\!', r'\?', '—', r'\.\.\.', r'\(', r'\)', r'\[', r'\]', r'<', r'>']
EXCLUDED_MARKS = ['«', '»', '—', '–', '"', '(', ')', '[', ']', '<', '>']
SPECIAL_SYMBOLS = [' ']

ALLOWED_SYMBOLS = set(list(ALPHABET) + PUNCTUATION_MARKS + SPECIAL_SYMBOLS + list(DIGITS)).difference(EXCLUDED_MARKS)

CORPUS_DIR = './corpus'
SOURCE_DIR_RAW = './corpus/raw'
TARGET_DIR_TOKENIZED = './corpus/tokenized'

if not os.path.exists(CORPUS_DIR):
    raise FileNotFoundError('Corpus directory not exists')

if not os.path.exists(TARGET_DIR_TOKENIZED):
    os.mkdir(TARGET_DIR_TOKENIZED)

file_names = os.listdir(SOURCE_DIR_RAW)
progress = 1

for file_name in file_names:
    with open(os.path.join(SOURCE_DIR_RAW, file_name), 'r') as file:
        raw_text = file.read()

    sentences = []
    sentence_start = 0
    for ending in re.finditer(f'({"|".join(SENTENCE_ENDING_MARKS_ESCAPED)})', raw_text):
        sentence_end = ending.end()
        sentence = raw_text[sentence_start: sentence_end]

        clear_sentence = ''
        for char in sentence:
            if char in ALLOWED_SYMBOLS:
                clear_sentence += char

        sentence = clear_sentence.strip().lower()

        if sentence.endswith(','):
            sentence = sentence[:-1]

        if sentence and sentence not in PUNCTUATION_MARKS:
            sentences.append(sentence)

        sentence_start = sentence_end

    with open(os.path.join(TARGET_DIR_TOKENIZED, file_name), 'w') as file:
        file.writelines(sentence + '\n' for sentence in sentences)

    print(f'\rProcessed texts: {progress} / {len(file_names)}', end='')
    progress += 1

print('\nCompleted.')
