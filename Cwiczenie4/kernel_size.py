from keras.datasets import mnist
from tensorflow.keras import *
import matplotlib.pyplot as plt
import numpy as np


def kernel_test(loops, epochs, X_train, y_train, X_test, y_test, kernel_size):
    accuracy_cnn = []
    loss_cnn = []

    for i in range(loops):
        model_cnn = Sequential([
            layers.Conv2D(32, kernel_size=kernel_size, activation='relu', input_shape=(28, 28, 1)),
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

    plt.figure("acc_cnn_ks")
    plt.plot(np.mean(accuracy_cnn, axis=0) * 100)
    plt.figure("loss_cnn_ks")
    plt.plot(np.mean(loss_cnn, axis=0))

    with open('kernel_size.txt', 'a+') as f:
        f.write(f'\nKernel {kernel_size}')
        f.write(f'\nA:\n{np.mean(accuracy_cnn, axis=0) * 100}')
        f.write(f'\nL:\n{np.mean(loss_cnn, axis=0)}\n')

    return np.mean(accuracy_cnn, axis=0) * 100, np.mean(loss_cnn, axis=0)

def EX_kernel_size(loops, epochs):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    kernel_test(loops, epochs, X_train, y_train, X_test, y_test, 2)
    kernel_test(loops, epochs, X_train, y_train, X_test, y_test, 3)
    kernel_test(loops, epochs, X_train, y_train, X_test, y_test, 4)
    kernel_test(loops, epochs, X_train, y_train, X_test, y_test, 8)
    kernel_test(loops, epochs, X_train, y_train, X_test, y_test, 14)


    plt.figure("acc_cnn_ks")
    plt.title('Convolutional Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Kernel 2x2',
                'Kernel 3x3',
                'Kernel 4x4',
                'Kernel 8x8',
                'Kernel 14x14',
                ])
    plt.ylim([85, 100])
    plt.savefig("ConvKernelSizeAcc.png")

    plt.figure("loss_cnn_ks")
    plt.title('Convolutional Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Kernel 2x2',
                'Kernel 3x3',
                'Kernel 4x4',
                'Kernel 8x8',
                'Kernel 14x14',
                ])
    plt.yscale('symlog', linthresh=0.1)
    plt.grid(True)
    plt.savefig("ConvKernelSizeLoss.png")
