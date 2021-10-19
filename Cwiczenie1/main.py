import random
import math


def activation(x_vector, w_vector):
    _sum = 0
    for i in range(len(x_vector)):
        _sum += (x_vector[i] * w_vector[i])
    return _sum


def threshold_function_bipolar(threshold, activation_value, is_biased=False):
    if is_biased:
        if activation_value >= 0:
            return 1
        return -1

    if threshold <= activation_value:
        return 1
    return -1


def threshold_function_unipolar(threshold, activation_value, is_biased=False):
    if is_biased:
        return int(activation_value >= 0)
    return int(threshold <= activation_value)


class Perceptron:

    def __init__(self, threshold, is_biased=False, is_unipolar=True):
        self.w_vector = []
        self.threshold = threshold
        self.is_biased = is_biased
        self.is_unipolar=is_unipolar

        if is_biased:
            self.w_vector = [threshold] + self.w_vector

    def predict(self, x_vector, add_bias=False):
        if add_bias:
            activation_value = activation([1] + x_vector, self.w_vector)
        else:
            activation_value = activation(x_vector, self.w_vector)

        if self.is_biased:
            if self.is_unipolar:
                return threshold_function_unipolar(self.w_vector[0], activation_value, self.is_biased)
            return threshold_function_bipolar(self.w_vector[0], activation_value, self.is_biased)

        if self.is_unipolar:
            return threshold_function_unipolar(self.threshold, activation_value, self.is_biased)
        return threshold_function_bipolar(self.threshold, activation_value, self.is_biased)

    def fit(self, X_matrix, desired_vector, learning_rate=0.01, weights_multiplier=0.5):
        needs_improvement = True
        X = X_matrix.copy()

        if self.is_biased:
            for i in range(len(X)):
                X[i] = [1] + X[i]

        self.w_vector = []
        for i in range(len(X[0])):
            self.w_vector += [(random.random() * random.choice([-1, 1])) * weights_multiplier]

        self.epochs = 0

        while needs_improvement and self.epochs < 10000:
            self.epochs += 1
            activation_values = []
            for x_vector in X:
                activation_values += [activation(x_vector, self.w_vector)]

            threshold_function_results = []
            for activation_value in activation_values:
                if self.is_unipolar:
                    threshold_function_results += [threshold_function_unipolar(self.threshold, activation_value, self.is_biased)]
                else:
                    threshold_function_results += [threshold_function_bipolar(self.threshold, activation_value, self.is_biased)]

            predictions_results = []
            for i in range(len(threshold_function_results)):
                predictions_results += [desired_vector[i] - threshold_function_results[i]]

            for i in range(len(self.w_vector)):
                weight_delta = 0
                for j in range(len(X)):
                    weight_delta += (learning_rate * predictions_results[j] * X[j][i])

                self.w_vector[i] += weight_delta
            needs_improvement = -2 in predictions_results or -1 in predictions_results or 1 in predictions_results or 2 in predictions_results

        # print("Number of epochs: " + str(self.epochs))

    def test(self, x_matrix, y_matrix_desired, is_verbal=True):
        correct = 0

        for i in range(len(x_matrix)):
            if self.predict(x_matrix[i], self.is_biased) == y_matrix_desired[i]:
                correct += 1
        if is_verbal:
            print(str(correct) + " out of " + str(len(x_matrix)) + " are correct (" + str(correct/len(x_matrix)*100) + "%)")
        return correct / len(x_matrix) * 100

class Adaline:
    def predict(self, x_vector):
        return threshold_function_bipolar(self.w_vector[0], activation([1] + x_vector, self.w_vector), True)

    def fit(self, X_matrix, desired_vector, learning_rate=0.01, weights_multiplier=0.5, max_error=0.3):
        X = X_matrix.copy()
        for i in range(len(X)):
            X[i] = [1] + X[i]

        self.w_vector = []
        for i in range(len(X[0])):
            self.w_vector += [(random.random() * random.choice([-1, 1])) * weights_multiplier]

        avg_error = max_error+1
        self.epochs = 0

        while max_error < avg_error and self.epochs < 10000:
            errors = []
            for k in range(len(X)):
                output = activation(X[k], self.w_vector)
                error = desired_vector[k] - output
                errors += [error * error]

                for i in range(len(self.w_vector)):
                    self.w_vector[i] += 2 * learning_rate * error * X[k][i]

            avg_error = sum(errors) / len(errors)
            self.epochs += 1
            # print(f"Error in epoch {self.epochs}: {avg_error}")

        # print("Numer of epochs: " + str(self.epochs))

    def test(self, x_matrix, y_matrix_desired, is_verbal=True):
        correct = 0

        for i in range(len(x_matrix)):
            if self.predict(x_matrix[i]) == y_matrix_desired[i]:
                correct += 1
        if is_verbal:
            print(str(correct) + " out of " + str(len(x_matrix)) + " are correct (" + str(correct/len(x_matrix)*100) + "%)")
        return correct/len(x_matrix)*100


def get_and_values(n_points=12, get_unipolar=True):
    n_points -= 4
    if n_points < 0:
        n_points = 0

    X = [[0,0], [0,1], [1,0], [1,1]]
    y = [0, 0, 0, 1]

    for i in range(n_points):
        j = random.randint(0, 3)
        X += [[X[j][0]+((random.random() - 0.5) * 0.1), X[j][1]+((random.random() - 0.5) * 0.1)]]
        y += [y[j]]

    if not get_unipolar:
        for i in range(len(y)):
            if y[i] == 0:
                y[i] = -1
    return X, y


def EX_perceptron_threshold(training_set_size=100):
    p = Perceptron(1, is_unipolar=False)
    p_results = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    p_accuracy = p_results.copy()
    thresholds = [100, 50, 25, 10, 5, 2.5, 1, 0.5, 0.1, 0.01, 0.001]

    for i in range(10):
        X, y = get_and_values(training_set_size, False)
        for j in range(len(p_results)):
            p.threshold = thresholds[j]
            p.fit(X, y, weights_multiplier=0.2)
            p_results[j] += p.epochs
            p_accuracy[j] += p.test(X, y, False)

    for i in range(len(p_results)):
        p_results[i] /= 10
        p_accuracy[i] /= 10

    print("\n- - - PERCEPTRON THRESHOLDS - - -")
    print(f"Checked values: {thresholds}")
    print("PERCEPTRON:\nEpochs: " + str(p_results) + "\nAccuracy: " + str(p_accuracy))


def EX_learning_rate(training_set_size=100):
    p = Perceptron(1, is_unipolar=False)
    a = Adaline()
    p_epochs = [0, 0, 0, 0, 0, 0, 0, 0]
    p_accuracy = p_epochs.copy()
    a_epochs = p_epochs.copy()
    a_accuracy = p_epochs.copy()
    learning_rates = [2, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]

    for i in range(10):
        X, y = get_and_values(training_set_size, False)
        for j in range(len(p_epochs)):
            p.fit(X, y, learning_rate=learning_rates[j])
            a.fit(X, y, learning_rate=learning_rates[j])

            p_epochs[j] += p.epochs
            a_epochs[j] += a.epochs
            p_accuracy[j] += p.test(X, y, False)
            a_accuracy[j] += a.test(X, y, False)

            if math.isnan(a.w_vector[0]) or math.isnan(a.w_vector[1]) or math.isnan(a.w_vector[2]):
                a_epochs[j] = math.inf

    for i in range(len(p_epochs)):
        p_epochs[i] /= 10
        a_epochs[i] /= 10
        p_accuracy[i] /= 10
        a_accuracy[i] /= 10

    print("\n- - - LEARNING RATE - - -")
    print(f"Checked values: {learning_rates}")
    print("PERCEPTRON:\n" + "Epochs: " + str(p_epochs) + "\nAccuracy: " + str(p_accuracy) + "\n")
    print("ADALINE:\n" + "Epochs: " + str(a_epochs) + "\nAccuracy: " + str(a_accuracy) + "\n")


def EX_weights(training_set_size=100):
    p = Perceptron(1, is_unipolar=False, is_biased=True)
    a = Adaline()
    p_epochs = [0, 0, 0, 0, 0, 0]
    p_accuracy = p_epochs.copy()
    a_epochs = p_epochs.copy()
    a_accuracy = p_epochs.copy()
    weight_multipliers = [1, 0.8, 0.6, 0.4, 0.2, 0]

    for i in range(1000):
        X, y = get_and_values(training_set_size, False)
        for j in range(len(p_epochs)):
            p.fit(X, y, weights_multiplier=weight_multipliers[j])
            a.fit(X, y, weights_multiplier=weight_multipliers[j])

            p_epochs[j] += p.epochs
            a_epochs[j] += a.epochs
            p_accuracy[j] += p.test(X, y, False)
            a_accuracy[j] += a.test(X, y, False)

    for i in range(len(p_epochs)):
        p_epochs[i] /= 1000
        a_epochs[i] /= 1000
        p_accuracy[i] /= 1000
        a_accuracy[i] /= 1000

    print("\n- - - WEIGHT MULTIPLIERS - - -")
    print(f"Checked values: {weight_multipliers}")
    print("PERCEPTRON:\n" + "Epochs: " + str(p_epochs) + "\nAccuracy: " + str(p_accuracy) + "\n")
    print("ADALINE:\n" + "Epochs: " + str(a_epochs) + "\nAccuracy: " + str(a_accuracy) + "\n")


def EX_perceptron_activation_f(training_set_size=100):
    pu = Perceptron(1, is_unipolar=True, is_biased=True)
    pb = Perceptron(1, is_unipolar=False, is_biased=True)
    pu_result = 0
    pb_result = 0

    for i in range(10):
        X, yu = get_and_values(training_set_size, True)
        yb = yu.copy()

        for k in range(len(yb)):
            if yb[k] == 0:
                yb[k] = -1

        pu.fit(X, yu)
        pb.fit(X, yb)

        pu_result += pu.epochs
        pb_result += pb.epochs

    pu_result /= 10
    pb_result /= 10

    print("\n- - - PERCEPTRON ACTIVATION - - -")
    print("PERCEPTRON UNIPOLAR: " + str(pu_result))
    print("PERCEPTRON BIPOLAR: " + str(pb_result))


def EX_adaline_error(training_set_size=100):
    a = Adaline()
    a_epochs = [0, 0, 0, 0, 0, 0, 0]
    a_accuracy = [0, 0, 0, 0, 0, 0, 0]
    error_values = [1, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1]

    for i in range(10):
        X, y = get_and_values(training_set_size, False)
        for j in range(len(error_values)):
            a.fit(X, y, max_error=error_values[j])
            a_epochs[j] += a.epochs

            a_accuracy[j] += a.test(X, y, False)

    for i in range(len(a_epochs)):
        a_epochs[i] /= 10
        a_accuracy[i] /= 10

    print("\n- - - ADALINE ERROR VALUES - - -")
    print(f"Checked values: {error_values}")
    print("Epochs:\n" + str(a_epochs))
    print("Accuracy:\n" + str(a_accuracy))


if __name__ == '__main__':
    # X, y = get_and_values(100, False)
    # X_test, y_test = get_and_values(32, False)
    #
    # print("- - - PERCEPTRON - - -")
    # p = Perceptron(threshold=0.5, is_biased=False, is_unipolar=False)
    # p.fit(X, y, learning_rate=0.01, weights_multiplier=0.5)
    #
    # print("\nTest with training set:")
    # p.test(X, y)
    #
    # print("\nTest with validation set:")
    # p.test(X_test, y_test)
    # print(p.w_vector)
    #
    # print("\n- - - ADALINE - - -")
    # a = Adaline()
    # a.fit(X, y, max_error=0.3, learning_rate=0.01, weights_multiplier=0.5)
    #
    # print("\nTest with training set:")
    # a.test(X, y)
    #
    # print("\nTest with validation set:")
    # a.test(X_test, y_test)
    # print(a.w_vector)

    EX_perceptron_threshold()
    EX_weights()
    EX_learning_rate()
    EX_perceptron_activation_f()
    EX_adaline_error()
