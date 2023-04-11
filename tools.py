from typing import Callable, Generator, Any, Iterable
import numpy as np

BOS, EOS, PAD = '<bos>', '<eos>', '<pad>'


class ModelIO:
    def __init__(self, vocabulary: Iterable[str]):
        vocabulary = list(vocabulary)

        self.vocabulary = vocabulary + [BOS, EOS, PAD]
        self.input_length = None
        self.output_length = None

        self.symbol_to_index = {i: s for s, i in enumerate(self.vocabulary)}
        self.index_to_symbol = self.vocabulary

        self.START_SYMBOL = self.symbol_to_one_hot(BOS)
        self.STOP_SYMBOL = self.symbol_to_one_hot(EOS)

    def symbol_to_one_hot(self, symbol: str):
        vector = np.zeros(len(self.vocabulary))
        vector[self.symbol_to_index[symbol]] = 1.0
        return vector

    def one_hot_to_symbol(self, vector: np.ndarray):
        index = vector.reshape(vector.size).tolist().index(1.)
        return self.index_to_symbol[index]

    def prepare_input(self, input_sequence: str, output_sequence=''):
        if not self.input_length or not self.output_length:
            self.input_length = len(input_sequence)

        raw_source_inp = list(input_sequence) + [PAD] * (self.input_length - len(input_sequence)) + [EOS]
        source_input = tuple(map(self.symbol_to_one_hot, raw_source_inp))

        if output_sequence == '':
            return source_input

        raw_target_inp = [BOS] + list(output_sequence) + [PAD] * (self.output_length - len(output_sequence))
        target_input = tuple(map(self.symbol_to_one_hot, raw_target_inp))
        raw_target_out = list(output_sequence) + [PAD] * (self.output_length - len(output_sequence)) + [EOS]
        target_output = tuple(map(self.symbol_to_one_hot, raw_target_out))

        return source_input, target_input, target_output

    def decode_output(self, output_sequence: Iterable[np.ndarray]):
        decoded = ''
        for output in output_sequence:
            symbol = self.one_hot_to_symbol(output)
            decoded += symbol
        return decoded


def take(generator, n=10):
    i = 0
    for value in generator:
        if i == n:
            break
        yield value
        i += 1


def mini_batch(samples_factory: Callable[[], Generator[tuple[str, str], Any, None]], batch_size=10, sample_count=10):
    def mini_batch_factory():
        samples = take(samples_factory(), sample_count)

        count = 0
        batch = []
        for sample in samples:
            batch.append(sample)
            count += 1

            if count >= batch_size:
                yield batch
                count = 0
                batch = []

        if batch:
            yield batch

    return mini_batch_factory


def vectorize(batch_factory: Callable[[], Generator[list, Any, None]], vocabulary):
    model_io = ModelIO(vocabulary)

    def vectorized_batch_factory():
        batches = batch_factory()

        for batch in batches:
            max_input_len, max_output_len = 0, 0
            for input_sequence, output_sequence in batch:
                if len(input_sequence) > max_input_len:
                    max_input_len = len(input_sequence)
                if len(output_sequence) > max_output_len:
                    max_output_len = len(output_sequence)

            model_io.input_length = max_input_len
            model_io.output_length = max_output_len

            source_inputs = []
            target_inputs = []
            target_outputs = []
            for input_sequence, output_sequence in batch:
                source_input, target_input, target_output = model_io.prepare_input(input_sequence, output_sequence)

                source_inputs.append(source_input)
                target_inputs.append(target_input)
                target_outputs.append(target_output)

            source_input_batches = []
            for i in range(max_input_len + 1):
                source_input_batch = []
                for source_input in source_inputs:
                    source_input_batch.append(source_input[i])
                source_input_batches.append(np.array(source_input_batch))

            target_input_batches = []
            for i in range(max_output_len + 1):
                target_input_batch = []
                for target_input in target_inputs:
                    target_input_batch.append(target_input[i])
                target_input_batches.append(np.array(target_input_batch))

            target_output_batches = []
            for i in range(max_output_len + 1):
                target_output_batch = []
                for target_output in target_outputs:
                    target_output_batch.append(target_output[i])
                target_output_batches.append(np.array(target_output_batch))

            yield source_input_batches, target_input_batches, target_output_batches

    return vectorized_batch_factory
