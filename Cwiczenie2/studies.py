import mlp
import threading
import time
from keras.datasets import mnist
import os

def run_and_write_to_file(train_X, train_y, val_X, val_y, test_X, test_y,
                          hidden_schema,
                          learning_rate,
                          max_epochs,
                          max_epochs_wo_impr,
                          n_of_batches,
                          f_activation,
                          file_name,
                          repeat_n_times,
                          loc=0.1,
                          scale=0.1):

    f = open(file_name + ".txt", "w")
    f.write(f"Repats: {repeat_n_times}\n")
    f.write(f"Batch size: {len(train_y)/n_of_batches}\n")
    f.write(f"Training set size: {len(train_y)}\n")
    f.write(f"Validation set size: {len(val_y)}\n")
    f.write(f"Test set size: {len(test_y)}\n")

    _sum_seconds = 0
    _sum_accuracy = 0
    _sum_epochs = 0


    for i in range(repeat_n_times):
        f.write(f"\n\n- - - TEST {i + 1} - - -\n")
        nn = mlp.MLP(784, hidden_schema, 10, f_activation=f_activation)
        nn.randomize_weights_and_bias(loc, scale)

        start_time = time.time()
        nn.fit(train_X, train_y, val_X, val_y,
                max_epochs=max_epochs,
                max_epochs_wo_impr=max_epochs_wo_impr,
                n_of_batches=n_of_batches,
                lr=learning_rate)

        seconds = time.time() - start_time
        accuracy = mlp.test(nn, test_X, test_y)
        epochs = nn.last_epoch

        _sum_seconds += seconds
        _sum_accuracy += accuracy
        _sum_epochs += epochs

        f.write(str(f"Seconds: {seconds}\n"))
        f.write(str(f"Accuracy: {accuracy}\n"))

        for l in range(len(nn.log)):
            f.write(nn.log[l] + "\n")

    f.write(str(f"\n\n- - - AVERAGE - - -\n"))
    f.write(str(f"Seconds: {_sum_seconds/repeat_n_times}\n"))
    f.write(str(f"Accuracy: {_sum_accuracy/repeat_n_times}\n"))
    f.write(str(f"Epochs: {_sum_epochs/repeat_n_times}\n"))
    f.close()
    print(f"File closed: {file_name}.txt")

def EX_neurons_count(train_X, train_y, val_X, val_y, test_X, test_y):
    hidden_neurons_schemas = [
                              [10, 10],
                              [784, 784]
                              [530, 250],
                              [600, 400, 200],
                              [640, 480, 320, 160],
                              [160, 320, 480, 640],
                              [200, 400, 600],
                              [250, 530],
                              [600, 600],
                              [600, 600, 600],
                              [600, 600, 600, 600],
                              [400, 400],
                              [400, 400, 400],
                              [400, 400, 400, 400],
                              [200, 200],
                              [200, 200, 200],
                              [200, 200, 200, 200],
                              [400, 25],
                              [400, 100, 25],
                              [400, 100, 45, 25],
                              [25, 45, 100, 400],
                              [25, 100, 400],
                              [25, 400]
                              ]
    threads = []
    for i in range(len(hidden_neurons_schemas)):
        n_of_batches = len(train_y) / 25

        path = "HiddenNeurons"
        if not os.path.exists(path):
            os.makedirs(path)
        name = [str(n) for n in hidden_neurons_schemas[i]]
        file_name = str(f"./HiddenNeurons/{i}_{'-'.join(name)}")
        threads += [threading.Thread(target=run_and_write_to_file,
                                     args=(train_X, train_y, val_X, val_y, test_X, test_y,
                                           hidden_neurons_schemas[i],
                                           0.1,  # współczynnik uczenia
                                           150,  # maks epok
                                           5,  # epok bez poprawy
                                           n_of_batches,  # rozmiar batcha
                                           mlp.f_sigmoid,  # funkcja aktywacji
                                           file_name,  # nazwa pliku
                                           10     # powtórzenia
                                            ))]

        threads[i].start()
        print(f"thread {i} started")

    for t in threads:
        t.join()


def EX_learning_rate(train_X, train_y, val_X, val_y, test_X, test_y):
    learning_rate = [5, 2, 1, 0.1, 0.01, 0.001, 0.0001]
    threads = []
    for i in range(len(learning_rate)):
        n_of_batches = len(train_y) / 25

        path = "LearningRate"
        if not os.path.exists(path):
            os.makedirs(path)
        name = [str(n) for n in learning_rate]
        file_name = str(f"./LearningRate/{i}_{name[i]}")
        threads += [threading.Thread(target=run_and_write_to_file,
                                     args=(train_X, train_y, val_X, val_y, test_X, test_y,
                                           [550, 50],  # schemat sieci ukrytej
                                           learning_rate[i],  # współczynnik uczenia
                                           150,  # maks epok
                                           5,  # epok bez poprawy
                                           n_of_batches,  # rozmiar batcha
                                           mlp.f_sigmoid,  # funkcja aktywacji
                                           file_name,  # nazwa pliku
                                           10     # powtórzenia
                                            ))]

        threads[i].start()
        print(f"thread {i} started")

    for t in threads:
        t.join()


def EX_batch_size(train_X, train_y, val_X, val_y, test_X, test_y):
    batch_size = [1, 4, 10, 25, 50, 100, 1000, 10000]
    threads = []


    for i in range(len(batch_size)):
        n_of_batches = len(train_y) / batch_size[i]

        path = "BatchSize"
        if not os.path.exists(path):
            os.makedirs(path)
        name = [str(n) for n in batch_size]
        file_name = str(f"./BatchSize/{i}_{name[i]}_lr1")
        threads += [threading.Thread(target=run_and_write_to_file,
                                     args=(train_X, train_y, val_X, val_y, test_X, test_y,
                                           [550, 50],  # schemat sieci ukrytej
                                           1,  # współczynnik uczenia
                                           150,  # maks epok
                                           5,  # epok bez poprawy
                                           n_of_batches,  # rozmiar batcha
                                           mlp.f_sigmoid,  # funkcja aktywacji
                                           file_name,  # nazwa pliku
                                           10     # powtórzenia
                                            ))]

        threads[i].start()
        print(f"thread {i} started")

    for t in threads:
        t.join()


def EX_starting_weights(train_X, train_y, val_X, val_y, test_X, test_y):
    loc = [0, 0, 0, 0, 0, 0]
    scale = [2, 1, 0.5, 0.1, 0.01, 0.001]
    assert(len(loc) == len(scale))

    threads = []
    for i in range(len(loc)):
        n_of_batches = len(train_y) / 25

        path = "StartingWeights"
        if not os.path.exists(path):
            os.makedirs(path)
        name = [str(f"loc_{n1}-scale{n2}-relu+") for n1, n2 in list(zip(loc, scale))]
        file_name = str(f"./StartingWeights/{i}_{name[i]}")
        threads += [threading.Thread(target=run_and_write_to_file,
                                     args=(train_X, train_y, val_X, val_y, test_X, test_y,
                                           [550, 50],  # schemat sieci ukrytej
                                           0.01,  # współczynnik uczenia
                                           150,  # maks epok
                                           5,  # epok bez poprawy
                                           n_of_batches,  # rozmiar batcha
                                           mlp.f_ReLU,  # funkcja aktywacji
                                           file_name,  # nazwa pliku
                                           10,  # powtórzenia
                                           loc[i],
                                           scale[i]
                                            ))]

        threads[i].start()
        print(f"thread {i} started")

    for t in threads:
        t.join()


if __name__ == "__main__":
    (train_X, train_y), (test_X, test_y) = mnist.load_data()

    x_t, y_t = mlp.MNIST_dataset_transform(mlp.normalize_input(train_X[:10000]), train_y[:10000])
    x_v, y_v = mlp.MNIST_dataset_transform(mlp.normalize_input(test_X[:2000]), test_y[:2000])
    x_test, y_test = mlp.MNIST_dataset_transform(mlp.normalize_input(train_X[58000:]), train_y[58000:])

    EX_neurons_count(x_t, y_t, x_v, y_v, x_test, y_test)
    print("Neurons ended")

    EX_learning_rate(x_t, y_t, x_v, y_v, x_test, y_test)
    print("Learning rate ended")

    EX_batch_size(x_t, y_t, x_v, y_v, x_test, y_test)
    print("Batch size ended")

    EX_starting_weights(x_t, y_t, x_v, y_v, x_test, y_test)
    print("Starting weights ended")
