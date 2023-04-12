from misc import LearningMode
from dataset import build_dataset
from tools import mini_batch, vectorize, ModelIO

from model import EncodeDecoder

# ---------------- Hyper parameters ----------------

MINI_BATCH_SIZE = 2  # number of sequences in a batch
NUMBER_OF_SAMPLES = 10  # how many to take for training
HIDDEN_LAYER_SIZE = 60  # size of context vectors

EPOCHS = 10  # number of epochs

ETA = 0.001  # learning rate
THETA = 1.0  # gradient clipping
MU = 0.95  # momentum

# ---------------- Usage ----------------

dataset = build_dataset(mode=LearningMode.WORD, shuffle=False)

mini_batches = mini_batch(dataset.training_set, MINI_BATCH_SIZE, NUMBER_OF_SAMPLES)
training_batches = vectorize(mini_batches, dataset.corpus.vocabulary)

model_io = ModelIO(dataset.corpus.vocabulary)

ed = EncodeDecoder(epochs=EPOCHS, eta=ETA, theta=THETA, mu=MU)
ed.input_layer_size = len(model_io.vocabulary)
ed.hidden_layer_size = HIDDEN_LAYER_SIZE

# training
errors = ed.fit(training_batches)

# prediction
input_sequence = 'подорож'  # the word taken from the first text

model_input = model_io.prepare_input(input_sequence)
result = ed.predict(model_input, model_io.START_SYMBOL, model_io.STOP_SYMBOL)
predicted_sequence = model_io.decode_output(result)

print('Input sequence:', input_sequence)
print('Predicted sequence:', predicted_sequence)
