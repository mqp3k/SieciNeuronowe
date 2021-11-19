import numpy as np
import math
import time
from keras.datasets import mnist


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
    return np.sum(-np.log(y_output) * y_desired)


class MLP:
    def __init__(self, input_n, hidden_n, output_n, f_activation=f_sigmoid):
        self.input_n = input_n
        self.hidden_n = hidden_n
        self.output_n = output_n
        self.W = []
        self.B = []
        self.randomize_weights_and_bias()
        self.f_activation = f_activation

    def randomize_weights_and_bias(self, loc=0.0, scale=1.0):
        neurons_in_layers = [self.input_n] + self.hidden_n + [self.output_n]
        self.W = []
        self.B = []
        # Losowanie początkowych wag i biasów
        for l in range(len(self.hidden_n) + 1):
            self.W += [np.random.normal(loc=loc, scale=scale, size=(neurons_in_layers[l + 1], neurons_in_layers[l]))]
            self.B += [np.random.normal(loc=loc, scale=scale, size=(neurons_in_layers[l + 1], 1))]

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

    def backpropagation(self, y_output, y_des, lr=0.001, batch_size=1):
        assert(len(y_output) == len(y_des))

        # Zapisanie prawidłowych wartości wyjściowych w formie macierzowej
        y_desired = desired_value_to_vector(y_des)

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
        for l in range(len(self.W)):
            self.W[l] -= lr/batch_size * self.dW[l]
            self.B[l] -= lr/batch_size * np.sum(self.dB[l], axis=1).reshape(len(self.dB[l]), 1)

    def fit(self, train_X, train_y, test_X, test_y, max_epochs_wo_impr=5, n_of_batches=1, max_epochs=1000, lr=0.1):
        assert(train_X.shape[1] == len(train_y))
        assert(len(train_y) % n_of_batches == 0)
        epoch = 0
        epochs_without_improvement = 0
        self.log = []

        # Zapisanie prawidłowych wyjściowych wartości testowych w formie macierzowej,
        # wykorzystywane jest do obliczania funkcji straty
        test_y_zeros = desired_value_to_vector(test_y)

        # Zainicjowanie początkowych wartości funkcji straty, zapis początkowych wag i biasów
        best_L = np.inf
        current_L = negative_log_likelihood(self.feed_forward(test_X), test_y_zeros)
        best_W, best_B = self.get_weights_copy()

        # Wykonywanie epok do osiągnięcia maksymalnej liczby epok lub osiągnięcia warunku early stopping
        # Przyjęto założenie early stopping jako n epok bez poprawy funkcji błędu, ustawia parametry z najmniejszym błędem
        while epoch < max_epochs and epochs_without_improvement < max_epochs_wo_impr:
            epoch += 1
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
                self.backpropagation(self.output.T, batched_y[b], batch_size=len(batched_y[b]), lr=lr)

            current_L = negative_log_likelihood(self.feed_forward(test_X), test_y_zeros)
            # print(f"Epoch: {epoch} \tC_function: {current_L}")
            self.log += [f"Epoch: {epoch} \tC_function: {current_L}"]
            # print(f"Epoch: {epoch} \tC_function: {current_L}")

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


if __name__ == "__main__":
    np.set_printoptions(threshold=np.inf)

    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    nn = MLP(784, [200, 200], 10, f_activation=f_ReLU)
    nn.randomize_weights_and_bias(0, 0.1)

    x_t, y_t = MNIST_dataset_transform(normalize_input(train_X[:25000]), train_y[:25000])
    x_v, y_v = MNIST_dataset_transform(normalize_input(test_X), test_y)

    test(nn, x_v, y_v)

    start_time = time.time()
    nn.fit(x_t, y_t, x_v, y_v, max_epochs=50, n_of_batches=1000, lr=0.01)
    print("%s seconds" % (time.time() - start_time))

    print(f"Accuracy: {test(nn, x_v, y_v)}")
    print(nn.log)
