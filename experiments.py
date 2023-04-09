from misc import LearningMode
from dataset import build_dataset
from tools import mini_batch, vectorize, ModelIO

from model import EncodeDecoder

MINI_BATCH_SIZE = 10

dataset = build_dataset(mode=LearningMode.WORD, shuffle=False)

mini_batches = mini_batch(dataset.training_set, MINI_BATCH_SIZE)
training_batches = vectorize(mini_batches, dataset.corpus.vocabulary)

model_io = ModelIO(dataset.corpus.vocabulary)

ed = EncodeDecoder(epochs=50, eta=0.001, theta=1., mu=0.9)
ed.input_layer_size = len(model_io.vocabulary)
ed.hidden_layer_size = 40

errors = ed.fit(training_batches)

result = ed.predict(model_io.prepare_input('подорож'), model_io.START_SYMBOL, model_io.STOP_SYMBOL)

print(''.join(map(model_io.one_hot_to_symbol, result)))
