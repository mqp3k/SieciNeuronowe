import numpy as np
import time
from keras.datasets import mnist
from enum import Enum
import matplotlib.pyplot as plt


class UpdateMethod(Enum):
    STANDARD = 1
    MOMENTUM = 2
    MOMENTUM_NESTEROV = 3
    ADAGRAD = 4
    ADADELTA = 5
    ADAM = 6


class WeightsInit(Enum):
    NORMAL_DIST = 1
    XAVIER = 2
    HE = 3


def f_sigmoid(z, derivative=False):
    if derivative:
        return f_sigmoid(1 - f_sigmoid(z))
    return 1 / (1 + np.exp(-z))


def f_tanh(z, derivative=False):
    t = np.tanh(z)
    if derivative:
        return 1 - np.power(t, 2)
    return t


def f_ReLU(z, derivative=False):
    if derivative:
        np.divide(1, 1 + np.exp(-z))
    return np.log(1 + np.exp(z))


def softmax(x):
    numerator = np.power(np.e, x)
    denominator = np.sum(numerator, axis=0)
    return np.divide(numerator, denominator)


def negative_log_likelihood(y_output, y_desired):
    return -np.sum(np.log(y_output) * y_desired) / len(y_desired)


class MLP:
    def __init__(self, input_n, hidden_n, output_n, f_activation=f_sigmoid):
        self.input_n = input_n
        self.hidden_n = hidden_n
        self.output_n = output_n
        self.W = []
        self.B = []
        self.randomize_weights_and_bias()
        self.f_activation = f_activation

        self.last_dW = []
        self.last_dB = []
        self._sumW = 0
        self._sumB = 0

    def randomize_weights_and_bias(self, loc=0.0, scale=1.0, method=WeightsInit.NORMAL_DIST):
        neurons_in_layers = [self.input_n] + self.hidden_n + [self.output_n]
        self.W = []
        self.B = []
        # Losowanie początkowych wag i biasów
        for l in range(len(self.hidden_n) + 1):
            self.W += [np.random.normal(loc=loc, scale=scale, size=(neurons_in_layers[l + 1], neurons_in_layers[l]))]
            self.B += [np.random.normal(loc=loc, scale=scale, size=(neurons_in_layers[l + 1], 1))]

            if method == WeightsInit.XAVIER:
                xavier_multiplier = np.sqrt(2 / (neurons_in_layers[l + 1] + neurons_in_layers[l]))
                self.W[l] *= xavier_multiplier
                self.B[l] *= 0

            if method == WeightsInit.HE:
                he_multiplier = np.sqrt(2 / neurons_in_layers[l])
                self.W[l] *= he_multiplier
                self.B[l] *= 0

    def feed_forward(self, X):
        self.X = X
        self.Z = []
        self.A = [X]

        # Propagacja w przód wejścia X
        for l in range(len(self.hidden_n) + 1):
            # Obliczanie pobudzenia warstwy l
            self.Z += [np.dot(self.W[l], self.A[l]) + self.B[l]]

            # Obliczanie aktywacji warstwy lub bez
            if l == len(self.hidden_n):
                self.A += [softmax(self.Z[l])]
            else:
                self.A += [self.f_activation(self.Z[l])]

        # Ustawienie wyjścia na ostatnią warstwę aktywowaną funkcją softmax
        self.output = self.A[-1]
        return self.output.T

    def backpropagation(self, y_output, y_des, lr=0.001, batch_size=1, w_update=UpdateMethod.STANDARD):
        assert(len(y_output) == len(y_des))
        gamma = 0.9
        deltaW = []
        deltaB = []
        # Zapisanie prawidłowych wartości wyjściowych w formie macierzowej
        y_desired = desired_value_to_vector(y_des)

        self.pre_gradient_update(w_update, gamma, deltaW, deltaB, lr, batch_size)

        # Obliczenie błędu w ostatniej warstwie
        # Lista dB (delta B) odpowiada przyrostowi biasów w poszczególnych warstwach
        # Lista dW (delta W) odpowiada przyrostowi wag w poszczególnych warstwach, w stosunku do dB wykonywane jest
        # dodatkowe mnożenie przez macierz aktywacji warstwy poprzedniej (wynika z pochodnych dla wag i biasów)
        self.dB = [-(y_desired - y_output).T]
        self.dW = [np.dot(self.dB[0], self.A[-2].T)]
        for l in reversed(range(1, len(self.W))):
            self.dB = [np.dot(self.W[l].T, self.dB[0]) * self.f_activation(self.A[l])] + self.dB
            self.dW = [np.dot(self.dB[0], self.A[l - 1].T)] + self.dW

        # Aktualizacja wag i biasów
        self.post_gradient_update(w_update, gamma, lr, batch_size, deltaW, deltaB)
        self.last_dW = self.dW.copy()
        self.last_dB = self.dB.copy()

    def pre_gradient_update(self, w_update, gamma, deltaW, deltaB, lr, batch_size):
        if len(self.last_dW) > 0 and w_update == UpdateMethod.MOMENTUM_NESTEROV:
            for l in range(len(self.W)):
                deltaW += [gamma * (lr / batch_size) * self.last_dW[l]]
                deltaB += [gamma * (lr / batch_size) * np.sum(self.last_dB[l], axis=1).reshape(len(self.dB[l]), 1)]
                self.W[l] -= deltaW[l]
                self.B[l] -= deltaB[l]

    def post_gradient_update(self, w_update, gamma, lr, batch_size, deltaW, deltaB):
        if w_update == UpdateMethod.STANDARD:
            for l in range(len(self.W)):
                self.W[l] -= lr/batch_size * self.dW[l]
                self.B[l] -= lr/batch_size * np.sum(self.dB[l], axis=1).reshape(len(self.dB[l]), 1)

        # ADAGRAD
        elif w_update == UpdateMethod.ADAGRAD:
            neurons_in_layers = [self.input_n] + self.hidden_n + [self.output_n]
            epsilon = 1e-08

            if self.epoch == 1:
                self.GW = []
                self.GB = []
                self._sumW = []
                self._sumB = []

                for l in range(len(self.hidden_n) + 1):
                    self.GW += [np.zeros(shape=(neurons_in_layers[l + 1], neurons_in_layers[l]))]
                    self.GB += [np.zeros(shape=(neurons_in_layers[l + 1], 1))]
                    self._sumW += [0]
                    self._sumB += [0]

            for l in range(len(self.W)):
                # sumy elementów
                self._sumW[l] += np.power(np.sum(self.dW[l]), 2)
                self._sumB[l] += np.power(np.sum(self.dB[l]), 2)
                np.fill_diagonal(self.GW[l], self._sumW[l])
                np.fill_diagonal(self.GB[l], self._sumB[l])

                tetaW = -lr/np.sqrt(self.GW[l] + epsilon) * self.dW[l] / batch_size
                tetaB = -lr/np.sqrt(self.GB[l] + epsilon) * np.sum(self.dB[l], axis=1).reshape(len(self.dB[l]), 1) / batch_size

                self.W[l] += tetaW
                self.B[l] += tetaB


        # ADADELTA
        elif w_update == UpdateMethod.ADADELTA:
            gamma = 0.9
            epsilon = 1

            if self.epoch == 1:
                self.gradW = []
                self.gradB = []
                self.tetW = []
                self.tetB = []
                self.tetW_1 = []
                self.tetB_1 = []
                self.tetW_2 = []
                self.tetB_2 = []
                for l in range(len(self.hidden_n) + 1):
                    self.gradW += [0]
                    self.gradB += [0]
                    self.tetW += [0]
                    self.tetB += [0]
                    self.tetW_1 += [(self.dW[l] * lr) ** 2]
                    self.tetB_1 += [(self.dB[l] * lr) ** 2]
                    self.tetW_2 += [0]
                    self.tetB_2 += [0]

            for l in range(len(self.hidden_n) + 1):
                self.gradW[l] = gamma * self.gradW[l] + (1 - gamma) * (self.dW[l] ** 2)
                self.gradB[l] = gamma * self.gradB[l] + (1 - gamma) * (self.dB[l] ** 2)
                self.tetW_2[l] = self.tetW_1[l]
                self.tetB_2[l] = self.tetB_1[l]
                self.tetW_1[l] = self.tetW[l]
                self.tetB_1[l] = self.tetB[l]
                self.tetW[l] = gamma * np.abs(self.tetW_2[l]) + (1 - gamma) * (self.tetW_1[l] ** 2)
                self.tetB[l] = gamma * np.abs(self.tetB_2[l]) + (1 - gamma) * (self.tetB_1[l] ** 2)

            for l in range(len(self.W)):
                self.tetW[l] = -np.sqrt(self.tetW[l] + epsilon) / (np.sqrt(self.gradW[l] + epsilon)) * self.dW[l] / batch_size
                self.tetB[l] = -np.sqrt(self.tetB[l] + epsilon) / (np.sqrt(self.gradB[l] + epsilon)) * self.dB[l] / batch_size
                self.W[l] += self.tetW[l]
                self.B[l] += np.sum(self.tetB[l], axis=1).reshape(self.dB[l].shape[0], 1) / self.tetB[l].shape[1]

        # ADAM
        elif w_update == UpdateMethod.ADAM:
            neurons_in_layers = [self.input_n] + self.hidden_n + [self.output_n]
            beta1 = 0.9
            beta2 = 0.899
            epsilon = 1

            if self.epoch == 1:
                self.lastMw = []
                self.lastMb = []
                self.lastVw = []
                self.lastVb = []

                for l in range(len(self.hidden_n) + 1):
                    self.lastMw += [np.zeros(shape=(neurons_in_layers[l + 1], neurons_in_layers[l]))]
                    self.lastVw += [np.zeros(shape=(neurons_in_layers[l + 1], neurons_in_layers[l]))]
                    self.lastMb += [np.zeros(shape=(neurons_in_layers[l + 1], 1))]
                    self.lastVb += [np.zeros(shape=(neurons_in_layers[l + 1], 1))]

            self.Mw = []
            self.Mb = []
            self.Vw = []
            self.Vb = []

            for l in range(len(self.hidden_n)+1):
                self.lastMw[l] = beta1 * self.lastMw[l] + (1 - beta1) * self.dW[l]
                self.lastMb[l] = beta1 * self.lastMb[l] + (1 - beta1) * self.dB[l]
                self.lastVw[l] = beta2 * self.lastVw[l] + (1 - beta2) * (self.dW[l] ** 2)
                self.lastVb[l] = beta2 * self.lastVb[l] + (1 - beta2) * (self.dB[l] ** 2)

                self.Mw += [self.lastMw[l] / (1 - beta1 ** self.epoch)]
                self.Mb += [self.lastMb[l] / (1 - beta1 ** self.epoch)]
                self.Vw += [self.lastVw[l] / (1 - beta2 ** self.epoch)]
                self.Vb += [self.lastVb[l] / (1 - beta2 ** self.epoch)]

            for l in range(len(self.W)):
                self.W[l] -= lr / (np.sqrt(self.Vw[l]) + epsilon) * self.Mw[l]
                self.B[l] -= (lr / (np.sqrt(np.sum(self.Vb[l], axis=1)) + epsilon) * np.sum(self.Mb[l], axis=1)).reshape(len(self.dB[l]), 1)

        # Momentum
        elif w_update == UpdateMethod.MOMENTUM or w_update == UpdateMethod.MOMENTUM_NESTEROV:
            if len(self.last_dW) > 0:
                for l in range(len(self.W)):
                    if w_update == UpdateMethod.MOMENTUM_NESTEROV:
                        self.W[l] += deltaW[l]
                        self.B[l] += deltaB[l]

                    self.W[l] -= ((lr / batch_size) * self.dW[l]) + gamma * (lr / batch_size) * self.last_dW[l]
                    self.B[l] -= ((lr / batch_size) * np.sum(self.dB[l], axis=1).reshape(len(self.dB[l]), 1)) + \
                                 gamma * (lr / batch_size) * np.sum(self.last_dB[l], axis=1).reshape(self.last_dB[l].shape[0], 1)
            else:
                for l in range(len(self.W)):
                    self.W[l] -= lr / batch_size * self.dW[l]
                    self.B[l] -= lr / batch_size * np.sum(self.dB[l], axis=1).reshape(len(self.dB[l]), 1)

    def fit(self, train_X, train_y, test_X, test_y, max_epochs_wo_impr=5, n_of_batches=1, max_epochs=1000, lr=0.1, w_update=UpdateMethod.STANDARD):
        assert(train_X.shape[1] == len(train_y))
        assert(len(train_y) % n_of_batches == 0)
        epoch = 0
        epochs_without_improvement = 0
        self.log = []
        self.loss = []
        self.accuracy = []

        # Zapisanie prawidłowych wyjściowych wartości testowych w formie macierzowej,
        # wykorzystywane jest do obliczania funkcji straty
        test_y_zeros = desired_value_to_vector(test_y)

        # Zainicjowanie początkowych wartości funkcji straty, zapis początkowych wag i biasów
        best_L = np.inf
        current_L = negative_log_likelihood(self.feed_forward(test_X), test_y_zeros)
        current_accuracy = test(self, test_X, test_y)
        self.loss += [current_L]
        self.accuracy += [current_accuracy]
        best_W, best_B = self.get_weights_copy()

        # Wykonywanie epok do osiągnięcia maksymalnej liczby epok lub osiągnięcia warunku early stopping
        # Przyjęto założenie early stopping jako n epok bez poprawy funkcji błędu, ustawia parametry z najmniejszym błędem
        # while epoch < max_epochs and epochs_without_improvement < max_epochs_wo_impr:
        while epoch < max_epochs:
            epoch += 1
            self.epoch = epoch
            # Mieszanie zbioru uczącego
            permutation = np.random.permutation(len(train_y))
            dataset_X = train_X[:, permutation]
            dataset_y = train_y[permutation]

            # Podział na batche
            batched_X = np.hsplit(dataset_X, n_of_batches)
            batched_y = np.array_split(dataset_y, n_of_batches)

            # Aktualizacja wag dla każdego zestawu danych uczących
            for b in range(len(batched_X)):
                self.feed_forward(batched_X[b])
                self.backpropagation(self.output.T, batched_y[b], batch_size=len(batched_y[b]), lr=lr, w_update=w_update)

            current_L = negative_log_likelihood(self.feed_forward(test_X), test_y_zeros)
            current_accuracy = test(self, test_X, test_y)
            # print(f"Epoch: {epoch} \tC_function: {current_L}")
            self.log += [f"Epoch: {epoch} \tC_function: {current_L}"]
            self.loss += [current_L]
            self.accuracy += [current_accuracy]
            print(f"Loss: {current_L} at epoch {epoch}")

            # Aktualizacja najlepszych parametrów i warunków wczesnego zatrzymania
            if (best_L > current_L):
                best_L = current_L
                best_W, best_B = self.get_weights_copy()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1


        # Wyświetlenie informacji o powodzie zakończenia uczenia
        if not (epochs_without_improvement < max_epochs_wo_impr):
            self.log += [f"Early stop on epoch: {epoch}"]
        else:
            self.log += [f"Stopped after max epochs: {epoch}"]

        # Ustawienie w modelu najlepszych znalezionych wag i biasów
        self.W = best_W
        self.B = best_B
        self.last_epoch = epoch

    def get_weights_copy(self):
        W_copy = self.W.copy()
        B_copy = self.B.copy()
        for i in range(len(W_copy)):
            W_copy[i] = W_copy[i].copy()

        for i in range(len(B_copy)):
            B_copy[i] = B_copy[i].copy()

        return W_copy, B_copy

    def predict(self, X):
        # Propagacja w przód wejścia X
        out = self.feed_forward(X)
        # Wybór indeksu neuronu o najwyższej wartości aktywacji
        return [np.argmax(o) for o in out]


def MNIST_dataset_transform(x_mat, y):
    x_to_ret = np.array([x.reshape(28*28, 1) for x in x_mat]).T[0]
    return x_to_ret, y


def desired_value_to_vector(y_des):
    y_des_value = y_des
    y_desired = np.zeros(shape=(len(y_des_value), 10))
    for i in range(len(y_desired)):
        y_desired[i][y_des_value[i]] = 1
    return y_desired


def normalize_data(data):
    return np.nan_to_num((data - np.min(data)) / (np.max(data) - np.min(data)))


def normalize_input(X):
    newX = []
    for i in range(len(X)):
        newX += [normalize_data(X[i])]
    return newX


def test(network, x_set, y_set):
    _sum = 0
    out_y = network.predict(x_set)
    for i in range(len(out_y)):
        if out_y[i] == y_set[i]:
            _sum += 1
    return (_sum / len(out_y)) * 100


def write_log_to_file(nn, file_name, accuracy=0.0, time=0.0):
    with open(file_name, 'a+') as f:
        for l in range(len(nn.log)):
            f.write(nn.log[l] + "\n")
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Time: {time}")
        f.write("\n\n")


def test_and_save(x_t, y_t, x_v, y_v):
    file_name = 'init.txt'
    description = 'Sigmoid lr=0.01 std_dev=0.1'

    std_dev = 1
    w_init = WeightsInit.HE
    w_update = UpdateMethod.ADAM
    fun = f_sigmoid
    learning = 0.01
    rep = 10
    batches = 600
    max_e = 10

    loss = []
    accuracy = []

    for i in range(rep):
        nn = MLP(784, [530, 250], 10, f_activation=fun)
        nn.randomize_weights_and_bias(0, std_dev, w_init)
        nn.fit(x_t, y_t, x_v, y_v, max_epochs=max_e, n_of_batches=batches, lr=learning, w_update=w_update)
        loss += [nn.loss]
        accuracy += [nn.accuracy]

    with open(file_name, 'a+') as f:
        f.write(f'\n\n{description}\n')
        f.write(f"Accuracy: {calc_mean(accuracy)}\n")
        f.write(f"Loss: {calc_mean(loss)}\n")

    plt.plot(calc_mean(loss))
    plt.show()
    plt.plot(calc_mean(accuracy))
    plt.show()


def calc_mean(values_list):
    means = []
    for i in range(len(values_list[0])):
        _sum = 0
        counter = 0

        for j in range(len(values_list)):
            if not np.isnan(values_list[j][i]):
                _sum += values_list[j][i]
                counter += 1
        if counter != 0:
            means += [_sum / counter]
        else:
            means += [0]

    return means

def plot(title, xlabel, ylabel, data, legend, filename):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for d in data:
        plt.plot(d)

    plt.legend(legend)
    plt.savefig(filename)
    plt.show()

if __name__ == "__main__":
    np.set_printoptions(threshold=np.inf)
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    x_t, y_t = MNIST_dataset_transform(normalize_input(train_X), train_y)
    x_v, y_v = MNIST_dataset_transform(normalize_input(test_X), test_y)

    test_and_save(x_t, y_t, x_v, y_v)

    data_loss = [

    ]

    data_accuracy = [
        # [91.64, 95.69, 96.50666666666666, 96.99666666666667, 96.96, 97.05333333333333,
        #  96.94000000000001, 96.61333333333333, 96.11, 96.16333333333334],
        # [91.375, 95.39500000000001, 96.39000000000001, 96.91, 96.71, 97.255, 97.185, 96.97, 96.77,
        #  96.88499999999999],
        # [91.58500000000001, 95.67, 96.44, 96.745, 96.975, 97.155, 97.11000000000001, 96.89500000000001, 96.705,
        #  96.46000000000001]
        [92.245, 93.3, 94.365, 95.04, 95.34, 95.495, 95.645, 95.85999999999999, 96.18, 96.16499999999999],
        [91.56, 92.995, 94.14500000000001, 95.03999999999999, 95.82, 95.755, 95.64999999999999, 95.53, 95.485, 95.525],
        [91.745, 92.99000000000001, 94.06, 94.62, 95.29, 95.80000000000001, 95.67, 95.625, 95.53999999999999, 95.525]

    ]

    legend = ['Sigmoid best',
              'Xavier',
              'He'
              ]

    plot('Sigmoid - Accuracy', 'Epochs', 'Accuracy', data_accuracy, legend, 'sigmoid_best.png')
