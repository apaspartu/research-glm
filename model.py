import numpy as np
import funcs as fn
import sys

np.set_printoptions(threshold=sys.maxsize)


class EncoderLSTM:
    def forward(self, X, H_prev, C_prev, weights, biases):
        W_xi, W_hi, W_xf, W_hf, W_xo, W_ho, W_xc, W_hc = weights
        b_i, b_f, b_o, b_c = biases

        I_t = fn.sigmoid(np.matmul(X, W_xi) + np.matmul(H_prev, W_hi) + b_i)
        F_t = fn.sigmoid(np.matmul(X, W_xf) + np.matmul(H_prev, W_hf) + b_f)
        O_t = fn.sigmoid(np.matmul(X, W_xo) + np.matmul(H_prev, W_ho) + b_o)
        C_tilda_t = np.tanh(np.matmul(X, W_xc) + np.matmul(H_prev, W_hc) + b_c)
        C_t = np.multiply(F_t, C_prev) + np.multiply(I_t, C_tilda_t)
        H_t = np.multiply(O_t, np.tanh(C_t))

        return H_t, C_t


class DecoderLSTM:
    def __init__(self, initial_cell_state, initial_hidden_state):
        self.time_step = 0

        self.prev_losses = []
        self.prev_layers = []
        self.prev_cell_states = [initial_cell_state]
        self.prev_hidden_states = [initial_hidden_state]
        self.prev_inputs = []

    def forward(self, X, H_prev, C_prev, weights, biases, training=True):
        W_xi, W_hi, W_xf, W_hf, W_xo, W_ho, W_xc, W_hc, W_hy = weights
        b_i, b_f, b_o, b_c, b_y = biases

        A_t = np.matmul(X, W_xi) + np.matmul(H_prev, W_hi) + b_i
        B_t = np.matmul(X, W_xf) + np.matmul(H_prev, W_hf) + b_f
        Z_t = np.matmul(X, W_xo) + np.matmul(H_prev, W_ho) + b_o
        M_t = np.matmul(X, W_xc) + np.matmul(H_prev, W_hc) + b_c

        I_t = fn.sigmoid(A_t)
        F_t = fn.sigmoid(B_t)
        O_t = fn.sigmoid(Z_t)
        C_tilda_t = np.tanh(M_t)

        C_t = np.multiply(F_t, C_prev) + np.multiply(I_t, C_tilda_t)
        H_t = np.multiply(O_t, np.tanh(C_t))
        U_t = np.matmul(H_t, W_hy) + b_y

        Y_predicted = fn.softmax(U_t)

        if training:
            self.prev_layers.append((I_t, F_t, O_t, C_tilda_t))
            self.prev_cell_states.append(C_t)
            self.prev_hidden_states.append(H_t)
            self.prev_inputs.append(X)

        self.time_step += 1

        return C_t, H_t, Y_predicted

    def loss(self, y_predicted, y_observed):
        dL_dU_t = fn.d_cross_entropy(y_predicted, y_observed)
        self.prev_losses.append(dL_dU_t)
        return dL_dU_t

    def backward(self, weights, biases):
        W_xi, W_hi, W_xf, W_hf, W_xo, W_ho, W_xc, W_hc, W_hy = weights
        b_i, b_f, b_o, b_c, b_y = biases

        dL_dW_hy = np.zeros(shape=W_hy.shape)
        dL_db_y = np.zeros(shape=b_y.shape)

        dL_dW_xo = np.zeros(shape=W_xo.shape)
        dL_dW_ho = np.zeros(shape=W_ho.shape)
        dL_db_o = np.zeros(shape=b_o.shape)

        dL_dW_xf = np.zeros(shape=W_xf.shape)
        dL_dW_hf = np.zeros(shape=W_hf.shape)
        dL_db_f = np.zeros(shape=b_f.shape)

        dL_dW_xi = np.zeros(shape=W_xi.shape)
        dL_dW_hi = np.zeros(shape=W_hi.shape)
        dL_db_i = np.zeros(shape=b_i.shape)

        dL_dW_xc = np.zeros(shape=W_xc.shape)
        dL_dW_hc = np.zeros(shape=W_hc.shape)
        dL_db_c = np.zeros(shape=b_c.shape)

        for t in range(self.time_step):
            dL_dU_t = self.prev_losses[t]

            # --------------- Output layer ----------------

            dL_dW_hy_t = np.matmul(self.prev_hidden_states[t + 1].T, dL_dU_t)
            dL_db_y_t = dL_dU_t
            dL_dH_t = np.matmul(dL_dU_t, W_hy.T) # 10 x 100

            dL_dW_hy += dL_dW_hy_t
            dL_db_y += np.sum(dL_db_y_t, axis=0)

            # --------------- Output gate ----------------

            dL_dW_xo_t = 0
            dL_dW_ho_t = 0
            dL_db_o_t = 0
            for k in range(0, t + 1):
                O_k = self.prev_layers[k][2]

                dH_t_dH_k = np.ones(shape=W_ho.shape)
                for j in range(k + 1, t + 1):
                    O_j = self.prev_layers[j][2]
                    dH_j_dO_j = np.tanh(self.prev_cell_states[j + 1])
                    dO_j_dZ_j = O_j * (1 - O_j)
                    dZ_j_dH_prev = W_ho

                    dH_j_dH_j_prev = np.matmul(np.matmul(dH_j_dO_j.T, dO_j_dZ_j), dZ_j_dH_prev)

                    dH_t_dH_k = np.matmul(dH_t_dH_k, dH_j_dH_j_prev)

                dH_k_dO_k = np.tanh(self.prev_cell_states[k + 1])
                dO_k_dW_xo = np.matmul((O_k * (1 - O_k)).T, self.prev_inputs[k])
                dO_k_dW_ho = np.matmul((O_k * (1 - O_k)).T, self.prev_hidden_states[k])
                dO_k_db_o = O_k * (1 - O_k)

                dL_dW_xo_t += np.matmul(np.matmul(dH_k_dO_k, dO_k_dW_xo).T, np.matmul(dL_dH_t, dH_t_dH_k))
                dL_dW_ho_t += np.matmul(np.matmul(dH_k_dO_k, dO_k_dW_ho).T, np.matmul(dL_dH_t, dH_t_dH_k))
                dL_db_o_t += np.matmul(np.matmul(dH_k_dO_k, dO_k_db_o.T).T, np.matmul(dL_dH_t, dH_t_dH_k))

            dL_dW_xo += dL_dW_xo_t
            dL_dW_ho += dL_dW_ho_t
            dL_db_o += np.sum(dL_db_o_t, axis=0)

            # ---------------- Forget gate ----------------

            dL_dW_xf_t = 0
            dL_dW_hf_t = 0
            dL_db_f_t = 0
            for k in range(0, t + 1):
                I_k, F_k, O_k, C_tilda_k = self.prev_layers[k]

                dH_t_dH_k = np.ones(shape=W_hf.shape)
                for j in range(k + 1, t + 1):
                    I_j, F_j, O_j, C_tilda_j = self.prev_layers[j]
                    dH_j_dC_j = O_j * (1 - (np.tanh(self.prev_cell_states[j + 1]) ** 2))
                    dC_j_dF_j = self.prev_cell_states[j]
                    dF_j_dB_j = F_j * (1 - F_j)
                    dB_j_dH_prev = W_hf

                    dH_j_dH_j_prev = np.matmul((dH_j_dC_j * dC_j_dF_j).T, np.matmul(dF_j_dB_j, dB_j_dH_prev))

                    dH_t_dH_k = np.matmul(dH_t_dH_k, dH_j_dH_j_prev)

                dH_k_dF_k = O_k * (1 - (np.tanh(self.prev_cell_states[k + 1]) ** 2)) * self.prev_cell_states[k]
                dF_k_dW_xf = np.matmul((F_k * (1 - F_k)).T, self.prev_inputs[k])
                dF_k_dW_hf = np.matmul((F_k * (1 - F_k)).T, self.prev_hidden_states[k])
                dF_k_db_f = F_k * (1 - F_k)

                dL_dW_xf_t += np.matmul(np.matmul(dH_k_dF_k, dF_k_dW_xf).T, np.matmul(dL_dH_t, dH_t_dH_k))
                dL_dW_hf_t += np.matmul(np.matmul(dH_k_dF_k, dF_k_dW_hf).T, np.matmul(dL_dH_t, dH_t_dH_k))
                dL_db_f_t += np.matmul(np.matmul(dH_k_dF_k, dF_k_db_f.T).T, np.matmul(dL_dH_t, dH_t_dH_k))

            dL_dW_xf += dL_dW_xf_t
            dL_dW_hf += dL_dW_hf_t
            dL_db_f += np.sum(dL_db_f_t, axis=0)

            # ---------------- Input gate ----------------

            dL_dW_xi_t = 0
            dL_dW_hi_t = 0
            dL_db_i_t = 0
            for k in range(0, t + 1):
                I_k, F_k, O_k, C_tilda_k = self.prev_layers[k]

                dH_t_dH_k = np.ones(shape=W_hf.shape)
                for j in range(k + 1, t + 1):
                    I_j, F_j, O_j, C_tilda_j = self.prev_layers[j]
                    dH_j_dC_j = O_j * (1 - (np.tanh(self.prev_cell_states[j + 1]) ** 2))
                    dC_j_dI_j = C_tilda_j
                    dI_j_dA_j = I_j * (1 - I_j)
                    dA_j_dH_prev = W_hi

                    dH_j_dH_j_prev = np.matmul((dH_j_dC_j * dC_j_dI_j).T, np.matmul(dI_j_dA_j, dA_j_dH_prev))

                    dH_t_dH_k = np.matmul(dH_t_dH_k, dH_j_dH_j_prev)

                dH_k_dI_k = O_k * (1 - (np.tanh(self.prev_cell_states[k + 1]) ** 2)) * C_tilda_k
                dI_k_dW_xi = np.matmul((I_k * (1 - I_k)).T, self.prev_inputs[k])
                dI_k_dW_hi = np.matmul((I_k * (1 - I_k)).T, self.prev_hidden_states[k])
                dI_k_db_i = I_k * (1 - I_k)

                dL_dW_xi_t += np.matmul(np.matmul(dH_k_dI_k, dI_k_dW_xi).T, np.matmul(dL_dH_t, dH_t_dH_k))
                dL_dW_hi_t += np.matmul(np.matmul(dH_k_dI_k, dI_k_dW_hi).T, np.matmul(dL_dH_t, dH_t_dH_k))
                dL_db_i_t += np.matmul(np.matmul(dH_k_dI_k, dI_k_db_i.T).T, np.matmul(dL_dH_t, dH_t_dH_k))

            dL_dW_xi += dL_dW_xi_t
            dL_dW_hi += dL_dW_hi_t
            dL_db_i += np.sum(dL_db_i_t, axis=0)

            # ---------------- Input node ----------------

            dL_dW_xc_t = 0
            dL_dW_hc_t = 0
            dL_db_c_t = 0
            for k in range(0, t + 1):
                I_k, F_k, O_k, C_tilda_k = self.prev_layers[k]

                dH_t_dH_k = np.ones(shape=W_hf.shape)
                for j in range(k + 1, t + 1):
                    I_j, F_j, O_j, C_tilda_j = self.prev_layers[j]
                    dH_j_dC_j = O_j * (1 - (np.tanh(self.prev_cell_states[j + 1]) ** 2))
                    dC_j_dC_tilda_j = I_j
                    dC_tilda_j_dM_j = 1 - (C_tilda_j ** 2)
                    dM_j_dH_prev = W_hc

                    dH_j_dH_j_prev = np.matmul((dH_j_dC_j * dC_j_dC_tilda_j).T, np.matmul(dC_tilda_j_dM_j, dM_j_dH_prev))

                    dH_t_dH_k = np.matmul(dH_t_dH_k, dH_j_dH_j_prev)

                dH_k_dC_tilda_k = O_k * (1 - (np.tanh(self.prev_cell_states[k + 1]) ** 2)) * I_k
                dC_tilda_k_dW_xc = np.matmul((I_k * (1 - I_k)).T, self.prev_inputs[k])
                dC_tilda_k_dW_hc = np.matmul((I_k * (1 - I_k)).T, self.prev_hidden_states[k])
                dC_tilda_k_db_c = I_k * (1 - I_k)

                dL_dW_xc_t += np.matmul(np.matmul(dH_k_dC_tilda_k, dC_tilda_k_dW_xc).T, np.matmul(dL_dH_t, dH_t_dH_k))
                dL_dW_hc_t += np.matmul(np.matmul(dH_k_dC_tilda_k, dC_tilda_k_dW_hc).T, np.matmul(dL_dH_t, dH_t_dH_k))
                dL_db_c_t += np.matmul(np.matmul(dH_k_dC_tilda_k, dC_tilda_k_db_c.T).T, np.matmul(dL_dH_t, dH_t_dH_k))

            dL_dW_xc += dL_dW_xc_t
            dL_dW_hc += dL_dW_hc_t
            dL_db_c += np.sum(dL_db_c_t, axis=0)

        self.time_step = 0

        weight_gradients = [dL_dW_xi, dL_dW_hi, dL_dW_xf, dL_dW_hf, dL_dW_xo, dL_dW_ho, dL_dW_xc, dL_dW_hc, dL_dW_hy]
        bias_gradients = [dL_db_i, dL_db_f, dL_db_o, dL_db_c, dL_db_y]

        return weight_gradients, bias_gradients


class EncodeDecoder:
    def __init__(self,  epochs=1, random_state=1, eta=0.01, theta=1.0, mu=0.9):
        self.epochs = epochs  # number of epochs
        self.random_state = random_state  # initial random generator seed

        self.eta = eta  # learning rate
        self.theta = theta  # gradient clipping threshold
        self.mu = mu  # momentum

        self.input_layer_size = None
        self.hidden_layer_size = 100

        self.weights = None
        self.biases = None

        self.weight_velocities = None
        self.bias_velocities = None

        self.prediction_threshold = 20  # maximal length of generated sequence

    def fit(self, training_samples):
        W_xi = np.random.RandomState(self.random_state).normal(loc=0.0, scale=0.01, size=(self.input_layer_size, self.hidden_layer_size))
        W_hi = np.random.RandomState(self.random_state).normal(loc=0.0, scale=0.01, size=(self.hidden_layer_size, self.hidden_layer_size))
        W_xf = np.random.RandomState(self.random_state).normal(loc=0.0, scale=0.01, size=(self.input_layer_size, self.hidden_layer_size))
        W_hf = np.random.RandomState(self.random_state).normal(loc=0.0, scale=0.01, size=(self.hidden_layer_size, self.hidden_layer_size))
        W_xo = np.random.RandomState(self.random_state).normal(loc=0.0, scale=0.01, size=(self.input_layer_size, self.hidden_layer_size))
        W_ho = np.random.RandomState(self.random_state).normal(loc=0.0, scale=0.01, size=(self.hidden_layer_size, self.hidden_layer_size))
        W_xc = np.random.RandomState(self.random_state).normal(loc=0.0, scale=0.01, size=(self.input_layer_size, self.hidden_layer_size))
        W_hc = np.random.RandomState(self.random_state).normal(loc=0.0, scale=0.01, size=(self.hidden_layer_size, self.hidden_layer_size))
        W_hy = np.random.RandomState(self.random_state).normal(loc=0.0, scale=0.01, size=(self.hidden_layer_size, self.input_layer_size))

        b_o = np.random.RandomState(self.random_state).normal(loc=0.0, scale=0.01, size=(1, self.hidden_layer_size))
        b_f = np.random.RandomState(self.random_state).normal(loc=0.0, scale=0.01, size=(1, self.hidden_layer_size))
        b_i = np.random.RandomState(self.random_state).normal(loc=0.0, scale=0.01, size=(1, self.hidden_layer_size))
        b_c = np.random.RandomState(self.random_state).normal(loc=0.0, scale=0.01, size=(1, self.hidden_layer_size))
        b_y = np.random.RandomState(self.random_state).normal(loc=0.0, scale=0.01, size=(1, self.input_layer_size))

        self.weights = [W_xi, W_hi, W_xf, W_hf, W_xo, W_ho, W_xc, W_hc, W_hy]
        self.biases = [b_i, b_f, b_o, b_c, b_y]

        self.weight_velocities = [np.zeros(w.shape) for w in self.weights]
        self.bias_velocities = [np.zeros(b.shape) for b in self.biases]

        errors = []

        for epoch in range(self.epochs):
            print('Epoch:', epoch)

            batch_loss = 0.0
            for source_input, target_input, target_output in training_samples():
                mini_batch_size = source_input[0].shape[0]

                H_prev = np.zeros(shape=(mini_batch_size, self.hidden_layer_size))
                C_prev = np.zeros(shape=(mini_batch_size, self.hidden_layer_size))

                # ---------------- Encoder ----------------

                encoder = EncoderLSTM()

                for X in source_input:
                    # forward propagation of the encoder
                    C_prev, H_prev = encoder.forward(X, H_prev, C_prev, self.weights[:-1], self.biases[:-1])

                # ---------------- Decoder ----------------

                decoder = DecoderLSTM(C_prev, H_prev)

                time_step = 0  # time step of the decoder
                total_loss = 0  # sum of losses across all time steps
                for X, Y in zip(target_input, target_output):
                    # forward propagation of the decoder
                    C_prev, H_prev, Y_predicted = decoder.forward(X, H_prev, C_prev, self.weights, self.biases)

                    # calculate and save loss of the current step
                    decoder.loss(Y_predicted, Y)

                    total_loss += fn.cross_entropy(Y_predicted, Y)

                    time_step += 1

                # average of losses across time steps and across samples in a batch
                batch_loss += (total_loss / time_step) / mini_batch_size

                # backpropagation through time
                weight_gradients, bias_gradients = decoder.backward(self.weights, self.biases)

                # ---------------- Clip gradients ----------------

                clipped_weight_gradients = []
                for weight_gradient in weight_gradients:
                    clipped_weight_gradients.append(fn.clip_gradient(weight_gradient, self.theta))

                clipped_bias_gradients = []
                for bias_gradient in bias_gradients:
                    clipped_bias_gradients.append(fn.clip_gradient(bias_gradient, self.theta))

                # ---------------- Update weights and biases ----------------

                for i in range(len(self.weights)):
                    # incorporate momentum
                    self.weight_velocities[i] = self.mu * self.weight_velocities[i] + clipped_weight_gradients[i]
                    self.weights[i] = self.weights[i] - self.eta * self.weight_velocities[i]

                for j in range(len(self.biases)):
                    # incorporate momentum
                    self.bias_velocities[j] = self.mu * self.bias_velocities[j] + clipped_bias_gradients[j]
                    self.biases[j] = self.biases[j] - self.eta * self.bias_velocities[j]

            print(batch_loss)
            errors.append(batch_loss)

        print(errors)
        return errors

    def predict(self, input_sequence, start, stop):
        H_prev = np.zeros(shape=(1, self.hidden_layer_size))
        C_prev = np.zeros(shape=(1, self.hidden_layer_size))

        # ---------------- Encoder ----------------
        encoder = EncoderLSTM()

        for X in input_sequence:
            C_prev, H_prev = encoder.forward(X, H_prev, C_prev, self.weights[:-1], self.biases[:-1])

        # ---------------- Decoder ----------------
        decoder = DecoderLSTM(C_prev, H_prev)

        output_sequence = []

        target_input = start
        for _ in range(self.prediction_threshold):
            C_prev, H_prev, Y_predicted = decoder.forward(target_input, H_prev, C_prev, self.weights, self.biases, training=False)

            target_input = fn.one_hot(Y_predicted)
            output_sequence.append(target_input)

            if fn.argmax(Y_predicted) == fn.argmax(stop):
                break

        return output_sequence
