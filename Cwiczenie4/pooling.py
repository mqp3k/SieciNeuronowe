from keras.datasets import mnist
from tensorflow.keras import *
import matplotlib.pyplot as plt
import numpy as np


def no_pooling(loops, epochs, X_train, y_train, X_test, y_test, kernel_size):
    accuracy_cnn = []
    loss_cnn = []

    for i in range(loops):
        model_cnn = Sequential([
            layers.Conv2D(32, kernel_size=kernel_size, activation='relu', input_shape=(28, 28, 1)),
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

    plt.figure("acc_cnn")
    plt.plot(np.mean(accuracy_cnn, axis=0) * 100)
    plt.figure("loss_cnn")
    plt.plot(np.mean(loss_cnn, axis=0))

    with open('pooling.txt', 'a+') as f:
        f.write(f'\nNoPooling')
        f.write(f'\nA:\n{np.mean(accuracy_cnn, axis=0) * 100}')
        f.write(f'\nL:\n{np.mean(loss_cnn, axis=0)}\n')

    return np.mean(accuracy_cnn, axis=0) * 100, np.mean(loss_cnn, axis=0)


def max_pooling(loops, epochs, pooling_size, X_train, y_train, X_test, y_test, kernel_size):
    accuracy_cnn = []
    loss_cnn = []

    for i in range(loops):
        model_cnn = Sequential([
            layers.Conv2D(32, kernel_size=kernel_size, activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D(pool_size=pooling_size),
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

    plt.figure("acc_cnn")
    plt.plot(np.mean(accuracy_cnn, axis=0) * 100)
    plt.figure("loss_cnn")
    plt.plot(np.mean(loss_cnn, axis=0))

    with open('pooling.txt', 'a+') as f:
        f.write(f'\nMaxPooling {pooling_size}')
        f.write(f'\nA:\n{np.mean(accuracy_cnn, axis=0) * 100}')
        f.write(f'\nL:\n{np.mean(loss_cnn, axis=0)}\n')

    return np.mean(accuracy_cnn, axis=0) * 100, np.mean(loss_cnn, axis=0)


def avg_pooling(loops, epochs, pooling_size, X_train, y_train, X_test, y_test, kernel_size):
    accuracy_cnn = []
    loss_cnn = []

    for i in range(loops):
        model_cnn = Sequential([
            layers.Conv2D(32, kernel_size=kernel_size, activation='relu', input_shape=(28, 28, 1)),
            layers.AvgPool2D(pool_size=pooling_size),
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

    plt.figure("acc_cnn")
    plt.plot(np.mean(accuracy_cnn, axis=0) * 100)
    plt.figure("loss_cnn")
    plt.plot(np.mean(loss_cnn, axis=0))

    with open('pooling.txt', 'a+') as f:
        f.write(f'\nAvgPooling {pooling_size}')
        f.write(f'\nA:\n{np.mean(accuracy_cnn, axis=0) * 100}')
        f.write(f'\nL:\n{np.mean(loss_cnn, axis=0)}\n')

    return np.mean(accuracy_cnn, axis=0) * 100, np.mean(loss_cnn, axis=0)

def EX_pooling(loops, epochs, kernel=4):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    no_pooling(loops, epochs, X_train, y_train, X_test, y_test, kernel)

    max_pooling(loops, epochs, (2, 2), X_train, y_train, X_test, y_test, kernel)
    max_pooling(loops, epochs, (4, 4), X_train, y_train, X_test, y_test, kernel)
    max_pooling(loops, epochs, (8, 8), X_train, y_train, X_test, y_test, kernel)

    avg_pooling(loops, epochs, (2, 2), X_train, y_train, X_test, y_test, kernel)
    avg_pooling(loops, epochs, (4, 4), X_train, y_train, X_test, y_test, kernel)
    avg_pooling(loops, epochs, (8, 8), X_train, y_train, X_test, y_test, kernel)

    plt.figure("acc_cnn")
    plt.title('Convolutional Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['No pooling',
                'Max pooling 2x2',
                'Max pooling 4x4',
                'Max pooling 8x8',
                'Avg pooling 2x2',
                'Avg pooling 4x4',
                'Avg pooling 8x8', ])
    plt.ylim([85, 100])
    plt.savefig("ConvPoolingAcc.png")

    plt.figure("loss_cnn")
    plt.title('Convolutional Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['No pooling',
                'Max pooling 2x2',
                'Max pooling 4x4',
                'Max pooling 8x8',
                'Avg pooling 2x2',
                'Avg pooling 4x4',
                'Avg pooling 8x8', ])
    plt.yscale('symlog', linthresh=0.1)
    plt.grid(True)
    plt.savefig("ConvPoolingLoss.png")
