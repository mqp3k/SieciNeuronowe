from keras.datasets import mnist
from tensorflow.keras import *
import matplotlib.pyplot as plt
import numpy as np


def kernel_test(loops, epochs, X_train, y_train, X_test, y_test, kernel_count=32):
    accuracy_cnn = []
    loss_cnn = []

    for i in range(loops):
        model_cnn = Sequential([
            layers.Conv2D(kernel_count, kernel_size=8, activation='relu', input_shape=(28, 28, 1)),
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

    plt.figure("acc_cnn_kc")
    plt.plot(np.mean(accuracy_cnn, axis=0) * 100)
    plt.figure("loss_cnn_kc")
    plt.plot(np.mean(loss_cnn, axis=0))

    with open('kernel_count.txt', 'a+') as f:
        f.write(f'\nKernel {kernel_count}')
        f.write(f'\nA:\n{np.mean(accuracy_cnn, axis=0) * 100}')
        f.write(f'\nL:\n{np.mean(loss_cnn, axis=0)}\n')

    return np.mean(accuracy_cnn, axis=0) * 100, np.mean(loss_cnn, axis=0)

def EX_kernel_count(loops, epochs):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    kernel_test(loops, epochs, X_train, y_train, X_test, y_test, 4)
    kernel_test(loops, epochs, X_train, y_train, X_test, y_test, 8)
    kernel_test(loops, epochs, X_train, y_train, X_test, y_test, 16)
    kernel_test(loops, epochs, X_train, y_train, X_test, y_test, 32)
    kernel_test(loops, epochs, X_train, y_train, X_test, y_test, 64)


    plt.figure("acc_cnn_kc")
    plt.title('Convolutional Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Kernels - 4',
                'Kernels - 8',
                'Kernels - 16',
                'Kernels - 32',
                'Kernels - 64',
                ])
    plt.ylim([85, 100])
    plt.savefig("ConvKernelCountAcc.png")

    plt.figure("loss_cnn_kc")
    plt.title('Convolutional Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Kernels - 4',
                'Kernels - 8',
                'Kernels - 16',
                'Kernels - 32',
                'Kernels - 64',
                ])
    plt.yscale('symlog', linthresh=0.1)
    plt.grid(True)
    plt.savefig("ConvKernelCountLoss.png")
