from keras.datasets import mnist
from tensorflow.keras import *
import matplotlib.pyplot as plt
import numpy as np


def topology_test1(loops, epochs, X_train, y_train, X_test, y_test):
    accuracy_cnn = []
    loss_cnn = []

    for i in range(loops):
        model_cnn = Sequential([
            layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=(28, 28, 1)),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])

        model_cnn.compile(optimizer='adam',
                          loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                          metrics=['accuracy']
                          )

        stats_cnn = model_cnn.fit(X_train, y_train,
                                  validation_data=(X_test, y_test),
                                  epochs=epochs,
                                  shuffle=True,
                                  batch_size=500,
                                  use_multiprocessing=True)
        accuracy_cnn += [stats_cnn.history['val_accuracy']]
        loss_cnn += [stats_cnn.history['val_loss']]

    plt.figure("acc_cnn_t")
    plt.plot(np.mean(accuracy_cnn, axis=0) * 100)
    plt.figure("loss_cnn_t")
    plt.plot(np.mean(loss_cnn, axis=0))

    with open('topology.txt', 'a+') as f:
        f.write(f'\nTopology 1')
        f.write(f'\nA:\n{np.mean(accuracy_cnn, axis=0) * 100}')
        f.write(f'\nL:\n{np.mean(loss_cnn, axis=0)}\n')

    return np.mean(accuracy_cnn, axis=0) * 100, np.mean(loss_cnn, axis=0)


def topology_test2(loops, epochs, X_train, y_train, X_test, y_test):
    accuracy_cnn = []
    loss_cnn = []

    for i in range(loops):
        model_cnn = Sequential([
            layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=(28, 28, 1)),
            layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=(28, 28, 1)),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])

        model_cnn.compile(optimizer='adam',
                          loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                          metrics=['accuracy']
                          )

        stats_cnn = model_cnn.fit(X_train, y_train,
                                  validation_data=(X_test, y_test),
                                  epochs=epochs,
                                  shuffle=True,
                                  batch_size=500,
                                  use_multiprocessing=True)
        accuracy_cnn += [stats_cnn.history['val_accuracy']]
        loss_cnn += [stats_cnn.history['val_loss']]

    plt.figure("acc_cnn_t")
    plt.plot(np.mean(accuracy_cnn, axis=0) * 100)
    plt.figure("loss_cnn_t")
    plt.plot(np.mean(loss_cnn, axis=0))

    with open('topology.txt', 'a+') as f:
        f.write(f'\nTopology 1')
        f.write(f'\nA:\n{np.mean(accuracy_cnn, axis=0) * 100}')
        f.write(f'\nL:\n{np.mean(loss_cnn, axis=0)}\n')

    return np.mean(accuracy_cnn, axis=0) * 100, np.mean(loss_cnn, axis=0)


def topology_test3(loops, epochs, X_train, y_train, X_test, y_test):
    accuracy_cnn = []
    loss_cnn = []

    for i in range(loops):
        model_cnn = Sequential([
            layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=(28, 28, 1)),
            layers.AvgPool2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])

        model_cnn.compile(optimizer='adam',
                          loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                          metrics=['accuracy']
                          )

        stats_cnn = model_cnn.fit(X_train, y_train,
                                  validation_data=(X_test, y_test),
                                  epochs=epochs,
                                  shuffle=True,
                                  batch_size=500,
                                  use_multiprocessing=True)
        accuracy_cnn += [stats_cnn.history['val_accuracy']]
        loss_cnn += [stats_cnn.history['val_loss']]

    plt.figure("acc_cnn_t")
    plt.plot(np.mean(accuracy_cnn, axis=0) * 100)
    plt.figure("loss_cnn_t")
    plt.plot(np.mean(loss_cnn, axis=0))

    with open('topology.txt', 'a+') as f:
        f.write(f'\nTopology 2')
        f.write(f'\nA:\n{np.mean(accuracy_cnn, axis=0) * 100}')
        f.write(f'\nL:\n{np.mean(loss_cnn, axis=0)}\n')

    return np.mean(accuracy_cnn, axis=0) * 100, np.mean(loss_cnn, axis=0)


def topology_test4(loops, epochs, X_train, y_train, X_test, y_test):
    accuracy_cnn = []
    loss_cnn = []

    for i in range(loops):
        model_cnn = Sequential([
            layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=(28, 28, 1)),
            layers.AvgPool2D(pool_size=(2, 2)),
            layers.Conv2D(32, kernel_size=3, activation='relu'),
            layers.AvgPool2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])

        model_cnn.compile(optimizer='adam',
                          loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                          metrics=['accuracy']
                          )

        stats_cnn = model_cnn.fit(X_train, y_train,
                                  validation_data=(X_test, y_test),
                                  epochs=epochs,
                                  shuffle=True,
                                  batch_size=500,
                                  use_multiprocessing=True)
        accuracy_cnn += [stats_cnn.history['val_accuracy']]
        loss_cnn += [stats_cnn.history['val_loss']]

    plt.figure("acc_cnn_t")
    plt.plot(np.mean(accuracy_cnn, axis=0) * 100)
    plt.figure("loss_cnn_t")
    plt.plot(np.mean(loss_cnn, axis=0))

    with open('topology.txt', 'a+') as f:
        f.write(f'\nTopology 2')
        f.write(f'\nA:\n{np.mean(accuracy_cnn, axis=0) * 100}')
        f.write(f'\nL:\n{np.mean(loss_cnn, axis=0)}\n')

    return np.mean(accuracy_cnn, axis=0) * 100, np.mean(loss_cnn, axis=0)


def topology_test5(loops, epochs, X_train, y_train, X_test, y_test):
    accuracy_cnn = []
    loss_cnn = []

    for i in range(loops):
        model_cnn = Sequential([
            layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=(28, 28, 1)),
            layers.AvgPool2D(pool_size=(2, 2)),
            layers.Conv2D(32, kernel_size=3, activation='relu'),
            layers.AvgPool2D(pool_size=(2, 2)),
            layers.Conv2D(32, kernel_size=3, activation='relu'),
            layers.AvgPool2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])

        model_cnn.compile(optimizer='adam',
                          loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                          metrics=['accuracy']
                          )

        stats_cnn = model_cnn.fit(X_train, y_train,
                                  validation_data=(X_test, y_test),
                                  epochs=epochs,
                                  shuffle=True,
                                  batch_size=500,
                                  use_multiprocessing=True)
        accuracy_cnn += [stats_cnn.history['val_accuracy']]
        loss_cnn += [stats_cnn.history['val_loss']]

    plt.figure("acc_cnn_t")
    plt.plot(np.mean(accuracy_cnn, axis=0) * 100)
    plt.figure("loss_cnn_t")
    plt.plot(np.mean(loss_cnn, axis=0))

    with open('topology.txt', 'a+') as f:
        f.write(f'\nTopology 2')
        f.write(f'\nA:\n{np.mean(accuracy_cnn, axis=0) * 100}')
        f.write(f'\nL:\n{np.mean(loss_cnn, axis=0)}\n')

    return np.mean(accuracy_cnn, axis=0) * 100, np.mean(loss_cnn, axis=0)


def topology_test6(loops, epochs, X_train, y_train, X_test, y_test):
    accuracy_cnn = []
    loss_cnn = []

    for i in range(loops):
        model_cnn = Sequential([
            layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=(28, 28, 1)),
            layers.AvgPool2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(10, activation='softmax')
        ])

        model_cnn.compile(optimizer='adam',
                          loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                          metrics=['accuracy']
                          )

        stats_cnn = model_cnn.fit(X_train, y_train,
                                  validation_data=(X_test, y_test),
                                  epochs=epochs,
                                  shuffle=True,
                                  batch_size=500,
                                  use_multiprocessing=True)
        accuracy_cnn += [stats_cnn.history['val_accuracy']]
        loss_cnn += [stats_cnn.history['val_loss']]

    plt.figure("acc_cnn_t")
    plt.plot(np.mean(accuracy_cnn, axis=0) * 100)
    plt.figure("loss_cnn_t")
    plt.plot(np.mean(loss_cnn, axis=0))

    with open('topology.txt', 'a+') as f:
        f.write(f'\nTopology 2')
        f.write(f'\nA:\n{np.mean(accuracy_cnn, axis=0) * 100}')
        f.write(f'\nL:\n{np.mean(loss_cnn, axis=0)}\n')

    return np.mean(accuracy_cnn, axis=0) * 100, np.mean(loss_cnn, axis=0)


def EX_topology(loops, epochs):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    topology_test1(loops, epochs, X_train, y_train, X_test, y_test)
    topology_test2(loops, epochs, X_train, y_train, X_test, y_test)
    topology_test3(loops, epochs, X_train, y_train, X_test, y_test)
    topology_test4(loops, epochs, X_train, y_train, X_test, y_test)
    topology_test5(loops, epochs, X_train, y_train, X_test, y_test)
    topology_test6(loops, epochs, X_train, y_train, X_test, y_test)


    plt.figure("acc_cnn_t")
    plt.title('Convolutional Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['C-D-D',
                'C-C-D-D',
                'C-P-D-D',
                'C-P-C-P-D-D',
                'C-P-C-P-C-P-D-D',
                'C-P-D'
                ])
    plt.ylim([85, 100])
    plt.savefig("ConvTopologyAcc.png")

    plt.figure("loss_cnn_t")
    plt.title('Convolutional Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['C-D-D',
                'C-C-D-D',
                'C-P-D-D',
                'C-P-C-P-D-D',
                'C-P-C-P-C-P-D-D',
                'C-P-D'
                ])
    plt.yscale('symlog', linthresh=0.1)
    plt.grid(True)
    plt.savefig("ConvTopologyLoss.png")
